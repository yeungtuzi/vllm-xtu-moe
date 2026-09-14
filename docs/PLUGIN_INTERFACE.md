# 插件/宿主接口契约(v0.2)

> **这份文档存在的理由**
>
> `docs/UPSTREAM_DRIFT.md` 量化的是**代码漂移**(多少文件、多少 hunk)。
> 但 v0.2 过程中最贵的问题不是代码漂移,而是**行为漂移**:
> 同一个 `xiaotu_moe.so`、同一份请求,
> 在 fork 编排下 **26.81 ms/token**,在主线下曾达 **1225 ms/token(47×)**。
> 这 47× **在任何 `diff` 里都不出现** —— 它只在"宿主怎么调用引擎"里。
>
> 根因是一句话:**引擎的默认值是按 fork 的节奏标定的,而节奏是宿主的属性。**
> 把引擎从 fork 里拿出来、换个宿主,代码一行没改,默认值全部失配。
>
> 所以这些耦合必须**写下来**(本文)并**在启动时断言**(`scripts/check_mainline_env.sh`)。

---

## 1. 契约总表

| # | 旋钮 | 引擎默认 | **主线下的必需值** | 缺失/设错的后果 | 证据 |
|---|---|---|---|---|---|
| C1 | `XIAOTU_MOE_SPIN_IDLE_US` | `5000` µs | **`0`** | 每次调用后**所有** worker 自旋 5 ms;43 层×~3 相 ⇒ 池几乎不停转:worker **3242% CPU**(≈32 核/worker)、load **119**;自造的争抢又反过来拖慢调用线程 —— **正反馈**,解码从 37 ms 漂到 **1225 ms** | §352 / §355 |
| C2 | `XIAOTU_MOE_NSLICE_SMALL` | 开 | **`0`** | `small_batch_workers()` 用标称 8 MAC/cycle 估算,DS-V4 解码维度算出 `single_us=6292 µs`(真实百 µs 级,**高估约 50×**)⇒ `wlimit=59/nt=60` ⇒ `stride=1` ⇒ `worker_limit_` 门闸**完全失效**(没有 worker 去 park),且走 `parallel_for_limited` —— **那条路会挂死** | §354 / §356 |
| C3 | `XIAOTU_MOE_THREADS` | `min(hw,120)` | **`60`** | 退到 120 就超出本机甜蜜点(每 CCD 4–5 核 = 96–120 才能跑满 DDR5 通道,再加核只增竞争) | R1 / §44 |
| C4 | `XIAOTU_MOE_RANK_SPLIT` | `1` | **`1`(不要动)** | 设 `2`(强制 fork 式全 node 布局)实测**请求全部挂死** | §361 |
| C5 | CPU 权重 NUMA 放置 | 需宿主传真实 rank/world | **必须传真实 `tp`/`rank`** | 硬编码 `num_processes=1/process_id=0` ⇒ 两个 rank 都把分片/线程池铺到**全部 8 node / 24 CCD** ⇒ 每 node 承受双份 ⇒ `oom-kill: CONSTRAINT_MEMORY_POLICY, nodemask=0`,进程**静默死亡** | §344 |
| C6 | `numactl --interleave=all` | — | **必需** | vLLM 加载期每个 rank 装**全部 256 专家**(≈138 GB/worker,EP 不在加载期切分存储),是**未绑定**的 first-touch 分配;实测总空闲 863 GB 时 node 0/2 已用 185/193 GB | §345 |
| C7 | `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` | 无(插件不设=0) | 长预填充场景设 **1024** | `_gp_min > 0` 是 GPU 预填充分支的**唯一**开关;不设 ⇒ 非常驻层永远走 CPU | §340 |
| C8 | `--max-num-batched-tokens` ≥ 阈值 | — | **`MBT ≥ GP_MIN`** | 阈值语义是 `size(0) >= T`;`T > MBT` ⇒ 一个 chunk 永远到不了阈值 ⇒ GPU 预填充**静默失效** | §340 |
| C9 | 预填充形状**不被 CUDA 图捕获** | PIECEWISE(会捕获) | 需显式 `--cudagraph-capture-sizes <解码尺寸>` | 捕获时走的是 CPU 分支,重放永远重放该分支 ⇒ **GPU 预填充形同虚设**。上游原生解法:只列解码尺寸,对应 `CUDAGraphMode.FULL_DECODE_ONLY` | §340c |

---

## 2. 为什么"在 fork 上没问题"不等于"在主线没问题"

同一份引擎、同一份 `bench_lat.sh`、同机实测(§358):

| 配置 | `bench_lat` C=1 **TPOT** |
|---|---|
| fork + **我们的引擎**(`ENV=lkxtu`) | **26.81 ms** ✅(与 0.1.0 文档 26.11–26.40 吻合) |
| 主线 + 我们的插件 | **~1225 ms** ❌ |

而把**同一套 `[cd-timing]` 探针**分别装到两种编排上,MoE 回调链只差 **1.24×**
(`period` 0.794 → 0.981 ms/层),调用次数**完全相同**(10 pass ≈ 430 次/rank)⇒
**多出来的 ~11 s 整个在 MoE 回调链之外**(§360)。

三点定曲线进一步归因(§361):

| 配置 | 墙钟 |
|---|---|
| `FAKE_ALL`(MoE 完全跳过) | 33.6 ms/token(vLLM 侧地板**正常**) |
| `FAKE_CPU`(保留 D2H/H2D/派发,只跳 CPU 计算) | **0.91 s / 请求** ⇒ 拷贝+派发几乎不花时间 |
| 真 MoE | **12.86 s / 请求** ⇒ **~12 s 全是 CPU-MoE 的"算/Wait"** |
| fork + 我们的引擎 | 1.88 s / 请求 |

⇒ **病灶是一句话**:同一份算力,`投递 → 算完` 的延迟在 fork 下稳定,在主线下游走于 **37 ms 与 1225 ms** 之间。
C1+C2 能把它在**干净窗口**拉回 **37.7 ms/token**(worker CPU 145%、load 11)。

---

## 3. 两个必须记住的**测量**陷阱(否则会得出 30× 的错误结论)

1. **`wall / out_tokens` 把预填充算进去了。**
   必须用 `(wall - ttft) / max(n_out - 1, 1)`,或直接读 `vllm bench` 的 **TPOT**。
   (我因此把 fork 上"512-token prompt 168 ms/token"当成解码,实际解码只有 ~38 ms。§358b)

2. **`[cd-timing/async]` 的 `period`/`compute` 只反映"投递",不反映异步 worker 的真实计算。**
   异步路径下 host 回调投递完就返回,所以 `period × 43 = 42 ms/token` 与端到端 `1225 ms/token`
   可以**同时成立**。要量真实算力,用 `XIAOTU_MOE_FAKE_CPU`(分离"算力 vs 搬运")或
   在 `cpu_decode` 后插诊断性 `sync`。**隔离微基准也覆盖不到异步 worker**
   (`bench_cd_plumbing.py` 给 0.32 ms/层,服务里实测 2.07 ms/层)。

---

## 4. 一条命令的自检

```bash
bash scripts/check_mainline_env.sh              # 检查当前 shell
TAG=ml_gp1 bash scripts/check_mainline_env.sh   # 检查某个已启动实例的 .env 记录
```

`serve_mainline.sh` 已在启动前自动调用它(`CHECK=0` 跳过,`CHECK_STRICT=1` 不通过即拒绝启动)。
退出码 = 未通过的条数。

---

## 5. 还没解决的(诚实记录)

C1+C2 让**干净测量窗口**回到 37.7 ms/token,但**在 `bench_lat` 的 random-512/128-out 负载下
主线仍会退化到 ~1.24 s/token**,而 fork 在同一负载下是 26.81 ms。
已排除:引擎算力、调用次数、qlen 谱(只有 `{1,2,4,8,16}`,无 257)、拷贝/派发(`FAKE_CPU` 0.91 s)、
`RANK_SPLIT=2`(挂死)。

**下一轮的入口**:在同负载下抓"异步 `投递→算完` 延迟"的直方图(引擎侧记 `enqueue_ts → worker_done_ts`),
以及 `/proc/<worker>/task/*/stat` 的**每线程 CPU 时间** —— 判断是"一直慢"还是"被 vLLM 线程周期性抢核"。
