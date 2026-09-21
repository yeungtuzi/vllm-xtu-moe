# Changelog

本文件记录 `vllm-xtu-moe` 的显著变更。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

---

## [Unreleased]

### Fixed

- **GLM-5.3-Flash 长 prompt 首次请求触发 CUDA `illegal memory access`,导致引擎崩溃**
  (`EngineDeadError`,bench 侧表现为 `TTFT 0.00` + `ConnectionRefusedError`)。
  根因是 **FP8 GPU 预填充装配运行在未正确同步的旁路流上**。
  修复:FP8 装配的旁路流**默认关闭**,并新增捕获期保护。
  详见下方「事故复盘」。

### Changed

- `gp_side_stream_enabled()` 默认值由「开」改为**「关」**
  (`XIAOTU_GP_ASM_SIDE_STREAM=1` 仍可显式打开,仅供性能实验)。
  **代价:长 prompt TTFT +17%**(实测 89.8 s → 105.4–108.8 s),待用正确的流间同步换回。

### Added

- `gp_capturing()`(`vllm_xiaotu_moe/mixed_experts.py`):CUDA graph 捕获期判定;
  捕获期**强制不用旁路流**,落到「当前流 + 不建跨流事件」的保守路径。
- `XIAOTU_DEATH_DEBUG=1`(`vllm_xiaotu_moe/__init__.py`,默认关闭):给
  SIGTERM/SIGINT/SIGHUP/SIGUSR1/SIGUSR2 装全线程栈 dump + 心跳,用于排查
  「进程被静默杀掉」类问题。
- `scripts/bench_random_12cells.sh`:random 数据集 12 格基准(3 模型 × {128,16384} × {C=1,C=2}),
  含 KV cap 自动推算与**前置断言**、日志/pidfile 双判据的早退检测、显式 `PYTHONPATH`、
  清场与观测分离。
- `dev-docs/RND_CAMPAIGN_DIAGNOSIS.md`:本次事故的完整诊断记录(含所有走过的弯路)。

---

## 事故复盘:GLM-5.3-Flash 长 prompt 非法访存

### 症状

`random` 数据集 12 格 campaign 中,GLM 的 `L=16384` 两格 `completed=0/8`、`TTFT 0.00`;
服务端日志里 `illegal memory access` 出现 **12 次**,`EngineDeadError`。

### 触发条件(实测收敛)

必须先跑过 **`L=128 C=1` 与 `L=128 C=2` 两种 decode batch**(对应两张 CUDA graph),
之后的**第一次长 prompt** 才越界。单变量矩阵:

| 序列 | 结果 |
|---|---|
| `128:1:8` + 长 | ✅ 干净 |
| `128:2:8` + 长 | ✅ 干净 |
| `128:1:16` + 长 | ✅ 干净 |
| 长 + 长 | ✅ 干净 |
| **`128:1:8` + `128:2:8` + 长** | ❌ **崩**(`illegal=12`) |
| 同上 + `CUDA_LAUNCH_BLOCKING=1` | ✅ **干净** |
| 同上,长 prompt 降到 4096 | ❌ 崩 |

「`CUDA_LAUNCH_BLOCKING=1` 让故障完全消失」= **竞态**的典型指纹,不是确定性越界。

### 根因

报错表面位置是 `byte_transpose.py` 的 Triton 转置 kernel,但那是**粘性错误的浮出点**
(`load_binary` 是模块加载,根本不碰显存)。逐层排除后锁定:

**`mixed_experts.py` 的 `apply()` 把整个 FP8 装配(`gpu_prefill_fp8.py` 的
`kmajor_from_engine_shards_fp8`,内含 H2D `cudaMemcpy2DAsync` + 转置 + GEMM)
跑在 `with torch.cuda.stream(_side)` 的旁路流上,而该路径对 CUDA graph 捕获期
没有任何保护** —— 而 MXFP4 路径(`gpu_prefill.py`)有 **3 处**
(`wait_no_capture` / `wait_capture_done` / `is_current_stream_capturing`)。
这是两条路径之间唯一的结构性不对称。

**单变量证据**:仅把 `XIAOTU_GP_ASM_SIDE_STREAM` 置 `0`(其余完全不变),
同一序列即干净,且 `GPU prefill ACTIVE=84`(装配确实跑了 84 次,不是"没走这条路")。

### 修复

```python
# mixed_experts.py
def gp_side_stream_enabled() -> bool:
    return os.environ.get("XIAOTU_GP_ASM_SIDE_STREAM", "0") == "1"   # 默认关

def gp_capturing() -> bool:
    try:    return bool(torch.cuda.is_current_stream_capturing())
    except Exception: return False

# apply() 内
_side_ok = gp_side_stream_enabled() and not gp_capturing()
```

### 验证

不设任何 env,依赖新默认值,跑那个此前 **5/5 必崩** 的序列:

```
step1 L=128   C=1 N=8 : ok=8/8  TTFT 1221 ms    Δillegal=0
step2 L=128   C=2 N=8 : ok=8/8  TTFT 2022 ms    Δillegal=0
step3 L=16384 C=1 N=1 : ok=1/1  TTFT 105602 ms  Δillegal=0
```

### 代价

| | 长 prompt TTFT |
|---|---|
| 旁路流开(旧默认) | ~89.8 s |
| 旁路流关(修复后) | ~105.4–108.8 s |

**+17%**。旁路流原本就是把这段 H2D 藏到 attention 后面用的;关掉退化为串行。
**后续应实现正确的流间同步(事件/依赖),把重叠换回来,而不是长期依赖关闭。**

### 排查中走过的弯路(留存以免重蹈)

1. **把异步粘性错误的浮出点当成根因**:先误判为 Triton 转置越界,直到 `XIAOTU_GPF_TT=0`
   仍崩才排除。
2. **自己实验脚本的 `kill -9` 制造了假故障**:脚本收尾 `kill -9 $(nvidia-smi ...)`
   只杀 worker,存活的 EngineCore 把它记成 `Worker proc died unexpectedly`,
   被我误读为「空闲期静默死亡」,并据此追查了 SIGTERM / memlock / OOM 数轮 —— **全是假象**。
   判定方法:考察「全程不做任何 kill」的对照组。
3. **prefix cache 混淆**:早期收窄实验里两条长请求用了**相同 seed**,第二条命中
   prefix cache(TTFT 89.7 s → 8.3 s),根本没跑装配,实验无效。
4. **开关设了没生效**:`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` 会被 `serve_*` 脚本写入的
   env-bridge 文件**覆盖**,`GPU_PREFILL=0` 也不会映射到它。**任何开关都必须先在日志里
   确认实际生效值。**
