# xiaotu-moe vs lk-moe 计算性能对比测试


> ⚠️ **历史材料(来自 `xiaotu-moe` 时期)** — 本文描述的是独立引擎项目 **xiaotu-moe**
> (闭源 `lk_moe` 的开源重实现、在 `Lvllm`/`Lvllmds4-x` fork 里做 drop-in 替换),
> 该项目**已转私有、仅作参考**。本项目 **`vllm-xtu-moe`** 是 **vLLM 主线插件**,
> 只把该引擎源码作为内置计算内核使用;本文的 fork / `lk_moe` 路线**不代表本项目路线**。
> 项目边界与当前待办见 `docs/BACKLOG.md`。

**Compute-performance comparison: xiaotu-moe vs lk-moe**

---

## 1. 目标 / Objective

在**不运行完整大模型**的前提下，公平地测量两个 MoE 引擎(`xiaotu-moe` 与 `lk-moe`)在 vLLM 实际调用路径上的**纯计算(单层 MoE)性能**，并回答以下问题：

> Without running the full model, fairly measure the **pure compute (single-layer MoE)** performance of both engines on the exact interfaces vLLM actually invokes, and answer:

- 同样的形状/权重/路由配置下，谁更快？快多少?/ Which is faster and by how much?
- 差距如何随 **batch 大小**变化?/ How does the gap change with **batch size**?
- `cpu_decode` 的"几乎为 0"耗时究竟是不是真的快?/ Is `cpu_decode`'s near-zero host time real?

**公平性原则 / Fairness principle:**
- 只测 vLLM 真实调用的接口(**不推测二进制内部**,直接测 `routed_experts.py` 里的调用签名)。
- 两个引擎用**相同合成权重 + 相同分批/路由**,逐个 batch 交替测。
- 用 **min-of-iters** 计时(对调度噪声鲁棒),同时记录 median。

---

## 2. 被测接口 / Interfaces under test

仅覆盖 vLLM `routed_experts.py` 实际使用的两个 CPU 原语接口(两组引擎签名完全一致):

| 接口 / Interface | vLLM 中的签名 (routed_experts.py) | 语义 / Semantics |
|---|---|---|
| `cpu_prefill` | `engine.cpu_prefill(qlen, top_k, eids_ptr, wts_ptr, x_ptr, out_ptr)` | 同步计算,两引擎行为一致(apples-to-apples)/ synchronous in both engines |
| `cpu_decode` | `engine.cpu_decode(stream_ptr, qlen, top_k, x_ptr, eids_ptr, wts_ptr, out_ptr)` | **lk_moe 为异步/重叠,host 只测到 launch;xiaotu 为同步** → wall-clock 不可比 / lk_moe async/overlapped, xiaotu sync → NOT comparable on host wall-clock |

> `engine.gpu_prefill(...)` 是 lk_moe 独有、xiaotu 没有的 GPU 通路(见 §5),不属于本次 **CPU 对比**范围。

---

## 3. 方法学 / Methodology

### 3.1 环境与依赖 / Environment
- **测试环境(conda)**:`xiaotumoe-vllm`(cp312, torch 2.11.0+cu130, vllm 2.3.11, 含 `lk_moe`; 通过 `PYTHONPATH` 加载 `xiaotu_moe`)。
- **注意**:`lvllmds4-x` 是生产环境,**绝不用于测试**。/ Prod env is never touched.
- lk_moe 在 import 时就会 spin 起 192 线程的 NUMA worker 池;xiaotu 也有 NumaWorkPool —— 两个引擎都持久占线程。
- lk_moe 构造会为 `gpu_prefill` 预分配 GPU 镜像显存;用 `BENCH_GPU=2`(`CUDA_VISIBLE_DEVICES=2`)把 `gpu_id` 映射到空闲 GPU,避免落到正在服务的 GPU0/1 而 OOM。

### 3.2 运行命令 / Running it
```bash
# 公平同步对比:全路由 256,prefill 全 batch(含 decode 尺寸 1..64)+集中路由示例
cd /home/user/lvllm/xiaotu-moe

BENCH_GPU=2 \
PYTHONPATH=/home/user/lvllm/xiaotu-moe \
LVLLM_MOE_NUMA_ENABLED=1 \
/home/user/anaconda3/envs/xiaotumoe-vllm/bin/python \
  scripts/bench_vs_lkmoe.py \
  --dims deepseek_flash --mode prefill --small-prefill --csv /tmp/bench_fair.csv

# 只测某个引擎(隔离比较,出该引擎自身的绝对耗时)
... --only lk --mode prefill --csv /tmp/lk.csv
... --only xiaotu --mode decode --csv /tmp/xm.csv
```

### 3.3 命令行参数 / CLI reference
| 参数 / Flag | 说明 / Meaning |
|---|---|
| `--dims {deepseek_flash,small,large}` | 预设维度(E,H,I,K,GK)/ dimension presets |
| `--custom E,H,I,K` | 自定义维度 / custom dims |
| `--mode {prefill,decode,all}` | 测哪些接口 / which interfaces |
| `--concentrate N` | 只路由 N 个专家(贴近 grouped/NUMA 路径)/ routing concentration |
| `--only {lk,xiaotu}` | 只测单个引擎 / run a single engine |
| `--prefill-max` | 用 `gpu_prefill`相关 vLLM 标准 GPU 通路(见 §5)/ GPU path |
| `--small-prefill` | 把 prefill 扫描扩展到 decode 尺寸 batch 1..64(公平同步对比关键)/ extend prefill sweep to decode sizes (KEY to fair sync comparison) |
| `--csv PATH` | 写 CSV / write CSV |

### 3.4 预设维度 / Dimension presets
| 项目 / Item | E (专家) | H (hidden) | I (inter) | K (topk) | GK (共享组) | 备注 / Note |
|---|---|---|---|---|---|---|
| `deepseek_flash` | 256 | 4096 | 2048 | 6 | 32 | = DeepSeek-V4-Flash 部署尺寸 |
| `small` | 128 | 2048 | 1024 | 6 | 32 | 更省内存/6.4GB×2 |
| `large` | 256 | 5120 | 2560 | 6 | 32 | 更大负担 |

### 3.5 计时口径 / Timing
- 每个 (引擎, batch) 做 warmup + **min-of-iters** + median;报告 `min/median ms`、`us/token`、`MoE tok/s` 与 **`ratio = xiaotu.min / lk.min`**(<1 ⇒ xiaotu 更快)。
- min-of-iters 对 CPU 上的调度噪声鲁棒 / robust to scheduler noise.

---

## 4. 结果(公平同步 cpu_prefill)/ Results (fair synchronous cpu_prefill)

模型 `DeepSeek-V4-Flash`,路由 256/256 专家。数值 = `min ms/layer`;`ratio = xiaotu/lk`(<1 表示 xiaotu 更快)。

| batch | lk_moe (ms) | xiaotu (ms) | ratio xiaotu/lk |
|---|---|---:|---:|---:|
| 1  | 63.97  | 78.66  | 1.23  |
| 2  |  6.30  | 78.70  | **12.5** |
| 4  | 50.28* | 78.84  | 1.57* |
| 8  |  6.33  | 78.90  | **12.5** |
| 16 |  6.29  | 79.13  | **12.6** |
| 32 |  8.40  | 79.81  | 9.5 |
| 64 |  8.44  | 89.27  | 10.6 |
| 128| 12.67  | 112.21 | 8.9 |
| 256| 14.92  | 228.32 | 15.3 |
| 512| 25.43  | 397.49 | 15.6 |
| 1024|47.37  | 689.04 | 14.5 |
| 2048|84.93  | 1337.58| 15.7 |

\* `B=4` 的 lk_moe 值明显异常(单次负载尖峰侵入,相邻 batch 均为 ~6ms),按 median 判断为噪声。

### 4.1 主要发现 / Key findings

1. **xiaotu 的 `cpu_prefill` 全面慢于 lk_moe**:全 batch 上比率 1.2–15.7×,且**随 batch 增大差距迅速拉大**(128 → ~9×,512+ → ~15×)。/ xiaotu's sync prefill is uniformly slower; the gap widens strongly with batch size.

2. **最值得注意的解码尺寸发现**:xiaotu 的 `cpu_prefill` 在 batch 1..32 几乎与 batch 无关,存在一个 **~78ms 的固定开销下限**;因此在解码尺寸 batch 2..16 上,xiaotu 比 lk_moe **慢约 12.5×**。这一固定开销与此前 `cpu_decode` 观测到的 xiaotu ~78ms 恒定值一致,且在**同一个同步内核上测量**(可公平对比),可判为真实性能特征而非测量伪影(可能与 NUMA 工作池/分组屏障/启动开销有关)。/ Striking decode-size finding: xiaotu cpu_prefill has a ~78ms batch-independent floor at B=1..32 → at decode sizes (2..16) it is ~12.5x slower; consistent across cpu_prefill and cpu_decode and measured on the same sync kernel, so treated as a real characteristic, not an artifact.

3. **`cpu_decode` 的"0 耗时"是假象**:lk_moe 的 host 侧 ~0.01ms 只是 **launch 时间**(计算在 worker 上异步重叠);xiaotu 是同步、占满整段时间。两者 wall-clock **不可比**,故解码尺寸一律改走同步 `cpu_prefill` 做公平对比。/

4. **可复现性提示 / Reproducibility caveat**:机器上有其它租户持续吃 CPU,绝对比率**逐次运行之间有波动**(前期一次运行在 batch 128..2048 得到 3.4–9.3×)。方向性结论(如"xiaotu 更慢,且大 batch 差距拉大""解码尺寸存在 ~78ms 固定开销")稳定;建议把比率当作"指示性区间"而非精确确定值,并拿 `--csv` 落盘归档。

---

## 5. GPU 通路说明 / GPU path note

`engine.gpu_prefill(...)` 是 **lk_moe 独有**的显式 GPU 通路,xiaotu 没有等价物。因此本次 **CPU 对比不包含该接口**;GPU 侧的公平对比改用 **vLLM 标准 GPU MoE** 路径(`routed_experts.forward_modular/forward_monolithic` → `quant_method.apply/apply_monolithic`,零新增 CUDA),由现有 vLLM fork 的 GPU mass-prefill 改动承载。/+ `gpu_prefill` is lk-only; for the GPU side the fair baseline is vLLM's standard GPU MoE path (no new CUDA).

---

## 6. GPU prefill 对比:lk 手搓 CUDA vs vLLM 标准 GPU MoE / GPU prefill: lk hand-CUDA vs vLLM standard GPU MoE

在 §5 的 CPU 对比之外,按用户要求另做一组 **GPU prefill 对比**,回答:

> gpu_prefill 也要对比性能——lk-moe 的手搓 CUDA 与 vLLM 标准的 MoE 代码,性能差多少?我们的**复用**是否会损失性能?

结论先行:**复用 vLLM 标准 GPU MoE 完全不损失性能,反而比 lk 手搓 `gpu_prefill` 快约两个数量级(~60–120×)。**

### 6.1 被测两侧 / The two sides

| 侧 / Side | 实现 / Implementation | vLLM 中的调用路径 |
|---|---|---|
| **lk 手搓 / lk hand-CUDA** | `engine.gpu_prefill(hidden, out, ids, wts, qlen, top_k, stream)` | lk_moe 自身显式 GPU 通路(xiaotu 无此接口) |
| **vLLM 标准 / vLLM standard(我们的复用)** | `Mxfp4MoEMethod` → `moe_kernel.apply`(MarLinExperts, 模块化) | `routed_experts.forward_modular` → GPU mass-prefill 分支(零新增 CUDA) |

测试脚本:`scripts/bench_gpu_prefill.py`(`--dims deepseek_flash|small|large`,`--custom`/`--batches`/`--csv`)。

### 6.2 方法学 / Methodology

- **同一合成 MXFP4 权重**喂给两侧;同一预路由输入:`x` bf16 `(B,H)`、`ids` int32 `(B,K)`、`tw` fp32 `(B,K)`。
- lk 侧:`lk_moe.MOE_MXFP4(make_lk_cfg(use_gpu_prefill=True), ...)`(生产 DeepSeek-V4-Flash 实际用 `MOE_WNA16`,两类的 `gpu_prefill` 行为已单独验证为相同)。
- vLLM 侧:`select_mxfp4_moe_backend` → `convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(MARLIN,...)` → `make_mxfp4_moe_kernel`,再以与 fork 相同的模块化调用 `kernel.apply(...)`。在 A100(sm_80) 上 **MARLIN 是唯一可用 backend 且 is_monolithic=False**(模块化),即复用路径正是生产 fork 走的那条。
- 用 **CUDA event 计时**(`torch.cuda.Event`),min-of-iters,只测 on-stream compute(不叠 CPU launch 噪声)。
- 跑在**空闲 GPU2**(`BENCH_GPU=2` → `CUDA_VISIBLE_DEVICES=2`,import torch 前),避免与服务的 GPU0/1 竞争/OOM。

### 6.3 数据(deepseek_flash 尺寸)/ Data (deepseek_flash dims)

| batch | lk `gpu_prefill` (ms) | vLLM MARLIN (ms) | 比值 vLLM/lk |
|---|---|---|---|
| 128   | 341.620 | 2.821 | 0.008 |
| 256   | 354.907 | 3.052 | 0.009 |
| 512   | 355.538 | 3.194 | 0.009 |
| 1024  | 356.481 | 3.786 | 0.011 |
| 2048  | 355.984 | 5.540 | 0.016 |

### 6.4 关键发现 / Key findings

- **lk `gpu_prefill` 有 ~340–365ms 的固定每调用开销,且与 batch 无关**(128→2048 都在 ~355ms 附近;wall==event,证实非测量伪影)。`MOE_MXFP4` 与生产用的 `MOE_WNA16` 行为相同。它甚至比 lk 自己的 `cpu_prefill`(B2048≈85ms) 还慢。
- **vLLM 标准 GPU MoE(MARLIN MXFP4)为 2.8–5.5ms,并随 batch 真实缩放**(输出是真实计算)。
- 因此**复用毫不损失性能:反而 ~60–120× 更快**(比值为 0.008–0.016)。GPU 上 vLLM 的标准 MoE 通路是明显更优的选项,且我们零新增 CUDA、直接复用 vLLM 代码。

---

## 7. 运行协议(与主服务共存)/ Co-existence protocol with the live service

机器上有常驻 vLLM 主服务(port 8070)持续请求模型,且基准测试会跟它**竞争 CPU**:

> 按用户要求:启动基准前**停止对主服务的请求**,等一小会;基准后台跑完、**释放 CPU 后**再恢复请求主服务;后台测试期间**保持静默、不调用大模型 API**。

因此:
- 基准用**后台任务**跑(`run_in_background`),stdout 落日志文件。
- 用一个轻量进度脚本确认测试正常推进后再等待结果。
- `min-of-iters` 可在负载下给出有意义的比值,但应把结果当**区间**解读。

---

## 8. 产物 / Artifacts

| 文件 / File | 说明 / Meaning |
|---|---|
| `scripts/bench_vs_lkmoe.py` | 综合对比基准(本报告对应脚本) |
| `/tmp/bench_fair_prefill.log` | 公平同步 prefill 结果(§4 数据来源) |
| `/tmp/bench_diverse.log` | 全路由 256,cpu_prefill+both(早期运行) |
| `/tmp/bench_concentrated.log` | 集中 16 专家早期运行 |
| `scripts/bench_gpu_prefill.py` | GPU prefill 对比基准(lk 手搓 CUDA vs vLLM 标准 GPU MoE,§6) |
| `/tmp/bench_gpu_prefill.log` | GPU prefill 对比日志(§6 数据来源) |
| `/tmp/bench_gpu_prefill.csv` | GPU prefill 对比 CSV(batch \| lk `gpu_prefill` ms \| vLLM MARLIN ms \| ratio) |

---

*作者:大河马(dahema@me.com) 由 DeepSeek Harness 辅助* / *Author: Dahema (dahema@me.com), assisted by DeepSeek Harness.*
