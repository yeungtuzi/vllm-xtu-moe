# vllm-xtu-moe 0.1.0 — Release Notes

**发布形态**:开源 CPU-GPU 混合推理引擎(`xiaotu_moe`)+ mainline vLLM 插件(`vllm_xiaotu_moe`)
**发布日期**:2026-09-14
**许可**:Apache-2.0
**下载**:
* `vllm_xtu_moe-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl`(二进制 wheel,含 5 个 ISA 变体的引擎 `.so`)
* `vllm-xtu-moe-0.1.0.tar.gz`(完整源码包,`git archive`)

---

## 一句话

> 在 **2×A100-40GB + 2×EPYC 9654(192 核,无 AMX)+ 1.5 TB DRAM** 这类"廉价异构"机器上,
> 把 **DeepSeek-V4-Flash(MXFP4,43 层 MoE,256 专家 top-6)** 跑到
> **单流 35.1 tok/s、C=4 聚合 107.1 tok/s**,与同机专有闭源引擎 `lk_moe` 2.4.2 **打平并反超**;
> 全链路开源、可改造、可审计。

* **定位**:科研 / 教学 / 小型实体应用,以及**信息安全要求高、性能要求不极端、预算受限**的场景
  —— 对这类项目,"看得见、改得动、审得清"是硬需求,性能只要够用。
* **动机**:8×H100 级 GPU 服务器起步数百万、高端上千万,科研教学不现实;
  而把专家权重放在 CPU/DRAM、只把注意力/KV 放 GPU,不仅省一个数量级的钱,
  还让模型**可以被研究**(逐层逐专家统计、插桩、安全性评估)。

---

## 一、功能

### 1.1 引擎 `xiaotu_moe`(纯 C++,无 nvcc 依赖)

| 能力 | 说明 |
|---|---|
| **CPU MoE 内核** | AVX2 / AVX512 基础 / AVX512-VNNI / **AVX512-BF16** / scalar 五个 ISA 变体,导入时按 `/proc/cpuinfo` 自动选最高可用(本机 EPYC 9654 → `avx512_bf16`) |
| **量化格式** | MXFP4(E2M1 + E8M0 组 scale)、FP8、INT4、BF16 |
| **FP4 解码** | fp32 LUT + `vpermps`(每 32 权重 19 → **10 条指令**);数值与 bf16-LUT 路径**逐位相同** |
| **行分块** | me=1/2/3/4 专用路径 + 混合 4+2 / 4+3;权重对多行只解码一次 |
| **NUMA 分片** | 权重按 NUMA 节点 `mbind` 切片,每 rank 只读本地内存(实测 **128/128 rc=0**) |
| **跨 rank EP** | 部分和归约走 `/dev/shm` + 自旋 barrier(**不走 NCCL**:每层 4–6.5 ms → **~14 µs**) |
| **异步握手** | 常驻 worker + `cudaHostAllocMapped` flag + `cuStreamWriteValue32/WaitValue32`,取代 `cudaLaunchHostFunc`(**图捕获安全**) |
| **诊断开关** | `FAKE_ALL` / `FAKE_CPU` / `NO_HOSTFN`(三段成本分解)、`CD_TIMING`(每层 period/compute/rest)、`MOE_PROFILE`(引擎内分相位)、pool `SHARD-JOBDIAG` |

### 1.2 插件 `vllm_xiaotu_moe`(mainline vLLM)

* 以 `vllm.general_plugins` 入口注册,**不改 vLLM 上游核心文件**;
* 通过 `FusedMoEFactory` / `RoutedExperts` 把 MoE 层接到 CPU 引擎;
* 附带 `gpu_prefill.py`(分层 GPU 预填充)、`mixed_experts.py`(混合专家放置)、`ple_offload.py`。

### 1.3 正确性与可观测性

* `scripts/test_block23_equiv.py` 数值门禁:**`OK=7 BAD=1 (me=1 max_rel 1.873e-02)`**,与基线逐字相同;
* host-func 路径 vs 异步握手路径输出 **逐位相同(u32 全等比例 1.0000)**;
* 服务级 greedy 文本对比(temperature=0,5 个 prompt):**5/5 完全一致**;
* 专家权重与激活在主机侧经过 ⇒ **可直接插桩做安全审计 / 路由行为研究**。

---

## 二、性能(全部本机实测)

### 2.1 测试环境与协议

```
GPU:2×A100-40GB(SXM)   CPU:2×EPYC 9654  NUMA:8 节点(BIOS NPS=4)
内存:24 通道 DDR5-4800,1.5 TB;实测流式带宽 ~740 GB/s
模型:DeepSeek-V4-Flash-0731(MXFP4,43 层 MoE + 3 层 DSpark 草稿)

客户端:scripts/bench_lat.sh  L=256 / OUT=512 / N=8 / CS="1 2 4"
服务端:TP=2 / GPU_UTIL=0.80 / MAXLEN=8192 / SEQS=8 / MBT=256 / MINBATCH=0 /
       PREFETCH=1 / EAGER=0(CUDA Graph)/ THREADS=60 / SPEC=0 / RESIDENT=0-11(12 层常驻)
```

### 2.2 主结果:同机 / 同协议 / 同配置,**唯一差别 = 引擎**

| 并发 | 专有 `lk_moe` 2.4.2 | **vllm-xtu-moe 0.1.0** | 我们 / 参考 |
|---|---|---|---|
| **C=1 TPOT** | 23.11 ms(41.04 t/s) | **26.11–26.40 ms(34.8–35.1 t/s)** | 0.88× |
| **C=2 聚合** | 59.73 t/s | **63.6–64.4 t/s** | **1.07×** |
| **C=4 聚合** | 86.92 t/s | **106.2–107.1 t/s** | **1.23×** |
| C=4 / OUT=2048(长跑) | — | **128.15 t/s** | — |

* 步代价拟合 `TPOT(C) = F + C·V`:参考 `F=16.85 / V=6.26 ms`;我们 **`F=24.40 / V=1.70 ms`**
  ⇒ **每 token 的边际成本更优**,差距集中在每步固定开销。
* 纯 GPU 地板(`XIAOTU_MOE_FAKE_ALL=1`,同配置):C=1 **16.80 ms**、C=4 21.71 ms(=144 t/s 上限)。

### 2.3 C=1 的成本账(三段严格相加)

```
26.1 ms = 16.80(纯 GPU 地板)+ 31 层 × 0.24(CPU 引擎)+ ~1.6(拷发)
参考同机 23.11 ms ⇒ 剩余差距 3.0 ms 全在"CPU 引擎每层 54 µs"
```

### 2.4 从基线到 0.1.0 的优化历程

| 阶段 | 改动 | C=1 TPOT | C=4 聚合 |
|---|---|---|---|
| 起点 | 43 层 MoE 全在 CPU + `cudaLaunchHostFunc` | 37.33 ms | ~34 t/s |
| ① | + 5 层 GPU 常驻 | 33.82 ms | 55.2 |
| ② | + 11 层常驻 + PERMV FP4 解码 | 31.78 ms | 60.2 |
| ③ | + 12 层常驻 + **异步握手** | 28.13 ms | 100.2 |
| ④ | + 并行区发布自旋 2000→200 | 26.48 ms | 105.0 |
| ⑤ | + 每 CCD 5 核(`THREADS=60`) | **26.17 ms** | **107.3** |
| | **累计** | **−30%** | **+213%** |

### 2.5 每层开销(服务内实测)

| 指标 | 历史(43 层全 CPU) | **0.1.0** |
|---|---|---|
| `compute`(CPU MoE + EP) | 0.410 ms/层 | **0.236–0.249 ms/层** |
| `rest`(GPU+拷贝+握手)稳态最小 | 0.53–0.66 ms/层 | **0.260–0.265 ms** |
| 边际内存带宽(权重流式) | — | **692–768 GB/s**(机器上限 ~740) |

### 2.6 其他工作负载

| 场景 | 结果 |
|---|---|
| 预填充 8192(冷)/ 32768 | **982 / 1287 t/s**(参考同机 883 / 828) |
| 投机解码(自由文本) | 净亏(接受率 2.1–2.75 < 盈亏平衡 ~2.9);`SPEC=0` 为默认 |
| 投机解码(可预测负载) | code 3.56 → **1.22×**;counting 6.00 → **1.64×** |
| 稳定性 | harness 16,000 次调用 + 服务端 16,384 token 长跑:**0 hang / 0 ERROR / 0 WATCHDOG** |

---

## 三、安装

### 3.1 二进制 wheel(推荐)

```bash
# 环境要求:Linux x86_64 · CPython 3.12 · glibc >= 2.34 · CPU 支持 AVX2(最佳 AVX512-BF16)
pip install vllm_xtu_moe-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl
```

wheel 自带 5 个 ISA 变体的 `_xiaotu_moe_C_*.so`,**不需要编译器,也不需要 nvcc**。

### 3.2 从源码构建

```bash
tar xzf vllm-xtu-moe-0.1.0.tar.gz && cd vllm-xtu-moe-0.1.0

# 1) 编译引擎的 5 个 ISA 变体(只需 g++;CUDA 头/库用于 host 侧 API)
PYTHON=/path/to/venv/bin/python \
PYBIND11_INC=/path/to/pybind11/include \
  bash scripts/build_engine_variants.sh

# 2) 打包 / 安装
python -m pip wheel . --no-deps -w dist
pip install dist/vllm_xtu_moe-0.1.0-*.whl
```

### 3.3 验证

```bash
python -c "import xiaotu_moe; print(xiaotu_moe.load())"     # 应打印所选变体
python -c "import vllm_xiaotu_moe; print('plugin ok')"      # 需已装 mainline vLLM
```

### 3.4 复用 vLLM 侧编排(可选)

本版本的性能数字是在 **lk 编排链**(`Lvllmds4-x` 的编排:常驻层 / GPU 预填充 / CPU 快慢路径 / DSpark 投机)
上测得的,仓库内提供 `scripts/serve_lk_port.sh` 一键起服务。

> ⚠️ 若你的目标是把本引擎接到**你自己的 vLLM fork**,请使用 `vllm_xiaotu_moe` 插件路径
> (实测可对 mainline vLLM 正常 `import`);两者共用同一个引擎 `.so`。

---

## 四、参考运行命令

### 4.1 启动服务(出厂默认,最优配置)

```bash
ENV=/path/to/venv \
TAG=v010 TP=2 GPUS=0,1 GPU_UTIL=0.80 MAXLEN=8192 SEQS=8 \
MBT=256 MINBATCH=0 PREFETCH=1 EAGER=0 SPEC=0 RESIDENT=0-11 \
  bash scripts/serve_lk_port.sh
```

出厂默认已固化三项(无需任何环境变量):

| 项 | 默认 | 说明 |
|---|---|---|
| 异步握手 | **开** | `EXTRA_ENV="XIAOTU_MOE_ASYNC=0"` 可回退到 `cudaLaunchHostFunc`(慢得多) |
| 并行区发布自旋 | **200** | `EXTRA_ENV="XIAOTU_MOE_PUBLISH_SETTLE=2000"` 回退到保守值 |
| 引擎线程数 | **60**(每 rank 12 CCD = 5 核/CCD) | `THREADS=48` 可回退 |

### 4.2 测量

```bash
# 延迟/吞吐(固定协议:短输入 + 长输出,避免被 CPU 预填充污染)
PORT=8070 TAG=v010 L=256 OUT=512 CS="1 2 4" SERVER_TAG=v010 bash scripts/bench_lat.sh

# 每层 period / compute / rest 拆分
EXTRA_ENV="XIAOTU_CD_TIMING=1 XIAOTU_CD_TIMING_EVERY=3100" ...

# 数值门禁
XIAOTU_LAYER1_NPZ=fixtures/real_layer1_model.npz python scripts/test_block23_equiv.py

# 分钟级引擎微基准(不加载模型)
NENGINES=14 REP=150 QLEN=1 LK_THREADS=120 DEDUP=6 MODE=serial \
  python scripts/bench_cd_plumbing.py
```

### 4.3 长上下文(把常驻层降到 11,换回 KV)

```bash
... RESIDENT=0-10 ...        # 11 层常驻:KV 2.19 GiB
```

---

## 五、兼容性与已知限制

| 项 | 说明 |
|---|---|
| **wheel 平台** | `cp312-cp312-manylinux_2_34_x86_64`;仅 Linux x86_64,glibc ≥ 2.34 |
| **CPU** | AVX2 起可跑;实测最佳为 **AMD Zen4(AVX512-BF16)**。**不依赖 Intel AMX** |
| **GPU** | 本项目在 SM80(A100)上验证;引擎本身与 GPU 型号无关 |
| **KV 容量** | 12 层常驻只剩 **0.6 GiB / 34,858 token** ⇒ 只适合短上下文;长上下文请用 11 层 |
| **发布自旋** | `PUBLISH_SETTLE` 是第 184 轮为规避"丢票 ⇒ 看门狗 abort"加的安全措施;0.1.0 取 200(有界、保留保护),根因窗口未消除,已在 `NOTES §329d` 记录 |
| **预填充** | 参考配置 `MINBATCH=0` 会关闭 GPU 预填充(省显存),预填充走 CPU(~230 t/s);**量聚合吞吐必须"短输入+长输出"**,否则量到的是预填充 |
| **投机解码** | 自由文本净亏,不建议默认开;可预测负载有 1.2–1.6× 收益 |
| **上游化** | 当前是把编排链移植进 vLLM fork;插件路径已可对 mainline 使用,**完全的主线支持与简化 patch/安装是 0.2 的目标** |

---

## 六、工程资产(随源码包提供)

| 类别 | 路径 |
|---|---|
| 引擎源码 | `xiaotu_moe/csrc/moe/`、`xiaotu_moe/csrc/python_binding/binding.cpp` |
| 插件 | `vllm_xiaotu_moe/` |
| 构建/部署 | `scripts/build_engine_variants.sh`、`scripts/deploy_engine.sh` |
| 基准 | `scripts/bench_lat.sh`、`scripts/bench_cd_plumbing.py`、`scripts/bench_vs_lkmoe.py` |
| 正确性 | `scripts/test_block23_equiv.py`、`scripts/probe_greedy.py` |
| 文档 | `内部报告 REPORT_vllm-xtu-moe.md`(项目报告 + 插图)、`内部文档 PERFORMANCE_OPTIMIZATION.md`、`内部调优记录 NOTES.md`(§300–§332)、`内部纪律 IRON_RULES.md`(铁律 R1–R9)、`内部记录 TRIED_AND_REVERTED.md`(R1–R112)、`(内部) OPTIMIZATION_RETROSPECTIVE.md`(人机协作复盘) |

---

## 七、致谢

`lk_moe` / `Lvllmds4-x` 作者提供了本机唯一的性能标杆与完整的混合推理编排范式,
本项目的所有"同机同配置只换引擎"对照都建立在它之上。本项目与其无代码复用关系,仅作对照基准。
