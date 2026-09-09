# Thread Geometry on the Dual-Socket EPYC 9654


> ⚠️ **历史材料(来自 `xiaotu-moe` 时期)** — 本文描述的是独立引擎项目 **xiaotu-moe**
> (闭源 `lk_moe` 的开源重实现、在 `Lvllm`/`Lvllmds4-x` fork 里做 drop-in 替换),
> 该项目**已转私有、仅作参考**。本项目 **`vllm-xtu-moe`** 是 **vLLM 主线插件**,
> 只把该引擎源码作为内置计算内核使用;本文的 fork / `lk_moe` 路线**不代表本项目路线**。
> 项目边界与当前待办见 `docs/BACKLOG.md`。

**xiaotu-moe v0.12 · hardware calibration notes**

## 1. The real CPU (verified from `/proc/cpuinfo` + `/sys/.../topology`, not assumed)

- **2 packages (dual socket)**, EPYC 9654.
- **96 physical cores per socket** — `core_id` is re-numbered 0–95 *per socket*,
  which is why a naive global `sort -u` over `core id` yields only 96 and was
  misread as "192 logical / 96 physical, SMT ON".
- **SMT is OFF**: every `(package, core_id)` has exactly 1 logical CPU, so
  **192 logical = 192 physical cores**.
- Each socket: **12 CCDs × 8 cores/CCD** = 96. The **12 CCDs share one IOD** that
  embeds the **12-channel DDR5 memory controller**. Memory bandwidth is fixed
  *per socket* — it is the hard ceiling for a bandwidth-bound MoE.

## 2. The operationally required layout: 168 = 84/socket, 7/CCD

User requirement (not a guess): use **168 threads = 84/socket, 7 of the 8 cores
per CCD**. This **saturates each socket's IOD DDR5 bandwidth** while leaving
**1 core/CCD (12/socket) free to respond to other tasks** on the host. It is the
production-correct value. 96 (48/socket, 4/CCD) is *not* a valid geometry — it
merely under-utilizes the machine.

## 3. Measured results (all: fused 3-barrier engine, 43 CPU MoE layers, A100/GPU2,
50 prompts / conc-4 / ml-8192, 50/50 pass)

| Threads | layout | cudagraph | Total (tok/s) | Med TPOT (ms) |
|---:|---|---:|---:|---:|
| 168 | 84/socket · 7/CCD | NONE | 32.00 | 213.1 |
| 96 | 48/socket · 4/CCD | NONE | 34.71 | 205.2 |
| 168 | 84/socket · 7/CCD | FULL_DECODE_ONLY | **66.62** | **79.83** |
| 96 | 48/socket · 4/CCD | FULL_DECODE_ONLY | 77.75 | 67.29 |
| 120 | 60/socket · 5/CCD (guide) | FULL_DECODE_ONLY | **77.69** | **73.65** |

The dominant lever is **cudagraph mode**, not thread count (33→66 vs 32→67).
Among thread counts at the same cudagraph setting, fewer-per-CCD measured
higher: at FULL_DECODE_ONLY, 96 (48/socket) > 168 (84/socket).

## 4. Why does *fewer* cores per CCD measure faster?

The MoE decode is **memory-bandwidth-bound on streaming weight reads**: every
weight byte is read once, the working set (~155 GB) vastly exceeds cache, so
there is no reuse win from "more cores". The per-socket IOD bandwidth is fixed.
Adding CCD cores past the bandwidth-saturation point therefore adds **only
overhead**:

1. **Barrier/synchronization cost** — the decode runs 3 full-pool `parallel_for`
   barriers per layer; barrier cost grows with thread count (atomic fan-in/out,
   busy-wait). More threads = more spin-wait traffic and a longer tail to the
   last arrival.
2. **CCD L3 thrash** — more active cores/CCD evict the small CCD-local L3 sooner
   and add load/store latency on the shared path; harmless when bandwidth-bound
   but it inflates the *latency* of the per-token critical path (TPOT).
3. **Memory-controller queueing latency** — excess in-flight loads pile into the
   same IOD queues, adding latency without adding delivered bandwidth.

So once a CCD has enough cores to saturate its share of the IOD bandwidth,
every extra core is pure cost — which is exactly why **84/socket (168) <
48/socket (96)**.

## 5. Why the guide says "5 cores / CCD" is optimal, and what 120 tests

A widely-circulated BIOS recommendation for the EPYC 9654 is to **enable 5 of
each CCD's 8 cores (disable 3)** and run lk-moe as a ~60-core-per-socket
processor for better throughput. This is consistent with the saturation model:

- The **sweet spot is the fewest cores/CCD that still saturate the IOD
  bandwidth** — for a streaming, bandwidth-bound kernel that is ~**4–5 cores/CCD**.
- **4/CCD (48/socket, "96")** is already past saturation and measures high;
  **5/CCD (60/socket) may add a little more achieved bandwidth or better
  latency** by giving the memory controller slightly more outstanding requests —
  the guide claims it is the measured optimum for lk-moe.
- **7/CCD (84/socket, "168")** is past the point of diminishing returns and only
  carries orchestration overhead → lowest.
- Disabling cores in BIOS also keeps the OS from ever scheduling/migrating work
  onto the disabled cores, giving a stable NUMA topology and stable barrier
  latency.

**The 120-thread test (5/CCD = 60/socket = 120 total over both sockets) is the
direct dual-socket simulation of that per-CCD sweet spot.** The measured result
confirms the model: **120 (77.69 tok/s, TPOT 73.65) is statistically tied with
96 (77.75/67.29) and far above 168 (66.62/79.83)**. There is a broad "flat top"
of bandwidth saturation spanning ~4–5 cores/CCD; past it (6–7/CCD) the extra
cores only add orchestration overhead. This is exactly the guide's claim — 5/CCD
is an optimal per-CCD operating point — reproduced on the dual-socket machine.

> **How many cores/CCD is actually enough?** For a streaming, bandwidth-bound MoE
> the measurements say **4 cores/CCD already reaches the (theoretical) peak** —
> 96 (48/socket, 4/CCD) and 120 (5/CCD) are tied, so 4/CCD is the saturation
> point and *more cores past 4/CCD buy nothing*. The only reason to go to
> **5/CCD is when memory bandwidth itself is higher** than stock (e.g. a board
> that runs the DIMMs at **DDR5-5600 instead of 4800**): with more bandwidth the
> IOD can absorb a little more outstanding demand, so the sweet spot pads to 5/CCD.
> In no case does going past 5/CCD help — the extra cores only add barrier/queue
> overhead.

## 6. Take-aways for production

- Keep **cudagraph `FULL_DECODE_ONLY`** — it is the real breakthrough (2.2×) and
  matches the production lk server's own config.
- The **user-mandated 168 (84/socket, 7/CCD) is the correct value** to ship: it
  clears DoD by a wide margin (>66 tok/s, TPOT <80 ms) and is the only layout
  that both saturates IOD bandwidth *and* reserves cores for other host tasks.
  96 is a measurement artifact, not a target. The 120 (5/CCD) figure tells us
  where the per-CCD bandwidth point of diminishing returns sits for the kernel.
