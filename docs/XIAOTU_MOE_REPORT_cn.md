# xiaotu-moe — 闭源 lk_moe 的开源重实现


> ⚠️ **历史材料(来自 `xiaotu-moe` 时期)** — 本文描述的是独立引擎项目 **xiaotu-moe**
> (闭源 `lk_moe` 的开源重实现、在 `Lvllm`/`Lvllmds4-x` fork 里做 drop-in 替换),
> 该项目**已转私有、仅作参考**。本项目 **`vllm-xtu-moe`** 是 **vLLM 主线插件**,
> 只把该引擎源码作为内置计算内核使用;本文的 fork / `lk_moe` 路线**不代表本项目路线**。
> 项目边界与当前待办见 `docs/BACKLOG.md`。

**里程碑报告(v1.0)— 中英双语(EN/CN)。**

- 仓库:`xiaotu-moe`(Python 包 `xiaotu_moe`,原生扩展 `_xiaotu_moe_C_*`)
- 目标:作为闭源 `lk_moe` 二进制引擎在 `Lvllmds4-x` / `Lvllm` vLLM fork 内的即插即用替代
- 许可立场:基于 Apache-2.0,不复制 lk_moe 专有代码

---

## 1. 项目来源(Origin)

**lk_moe** 是闭源 CPU 推理引擎,以 `cp312 manylinux x86_64` wheel 发布在 PyPI(版本 2.3.3),无源码发行,许可是专有的(签署人 Qiong GU / guqiong9696@gmail.com,2025-11-08)。它被(同作者的)vLLM fork `Lvllm`、`Lvllmds4-x`、`lsglang` 用来把仅 GPU 的 MoE 前向替换为 **CPU–GPU 混合引擎**,从而让更大模型装入 显存+内存(即"1+1=2"、100% 显存利用方案)。

虽然 lk_moe 闭源,作者随包提供了 `THIRD_PARTY_LICENSES`,**正式点名三份衍生自 KVCache.AI ktransformers(Apache-2.0)的文件**:`csrc/lk_moe/moe.h`、`csrc/lk_moe/moe.cpp`、`csrc/lk_moe/backend_numa.cpp`。对 wheel 的逐字节检查确认:
- **NUMA 调度器**(`Backend_NUMA`)源自 `ktransformers_ext/cpu_backend/backend.cpp`(含作者原始拼写错误"suscess"的字符串铁证)。
- **MoE 骨架保留**,但**量化计算核心被作者彻底重写(V2)**为专有模板 `MOE_V2<WeightTraits, ActivationType>`,共 10 个实例化,并剥离了 ggml/llamafile 依赖。

**结论:** 重实现可行。调度器与 MoE 骨架开源可查(ktransformers / lktransformers,Apache-2.0);真正需要从零写的是量化 GEMM/反量化内核(FP8/BF16/FP16/WNA16/NVFP4/MXFP4)。完整谱系见 `LK_MOE_INVESTIGATION_REPORT.md`。

## 2. 目标(Goals)

1. 用开源可得的零件,打造闭源 `lk_moe` CPU MoE 引擎的**源码可见替代品**,不复制专有代码。
2. **ABI 兼容即插即用**:实现 `MOEConfigV2` + `data_ptr()` / `cpu_decode` / `cpu_prefill` 契约,使 fork 的 `moe_runner.py` 分派无需改动(`self.lk_moe` → `xiaotu_moe` 实例)。
3. 在真实模型上达到与 lk_moe 的**正确性对齐**,用真实 DeepSeek-V4-Flash-0731 checkpoint 验证。
4. **约束**:复用现有代码;`MOEConfigV2` 布局由重实现方确定;包名 `xiaotu-moe`;不依赖 ggml/llama.cpp/libnuma。

## 3. 思路(Approach)

分阶段实施,每阶段均可独立对照闭源二进制验证:

- **Phase 1** — 最小闭环:BF16 CPU MoE(无依赖的 `MOE_V2<WeightTraits, ActivationType>` + bf16 gemm + 小 `std::thread` 池)。确立编排语义(gate → up → SiLU → down → 加权累加)。
- **Phase 2** — 扩展到量化类型 FP8 / WNA16 / MXFP4,每种都放在 CRTP `WeightTraits` 接口之后,使编排循环逐字共享。
- **Phase 3a** — NVFP4(策略 A:复用共享 E2M1 内核 + 每专家 global scale,对齐 vLLM NVFP4 语义)。
- **Phase 3b** — **关键架构决策:xiaotu-moe 纯 CPU**;GPU 侧完全复用 fork 的 vLLM 标准 GPU MoE(不手写 CUDA)。作者自写 GPU CUDA 文件(`moe_v2_gpu_*.cu`)被理解为**两个角色**——(1) `cpu_decode` 所需的浅层流/缓冲桥接(始终需要,xiaotu-moe 通过 Python `_cpu_decode` 契约对接、无需 CUDA 核);(2) 深层自写 `gpu_prefill` 核,是 vLLM 标准 GPU MoE 的冗余替代,**不重写、不复制**。
- **Phase 4** — 打包:多 ISA `.so` 变体 + `_dynamic_loader` 运行时选择 + 集成进 fork 并端到端验证。

## 4. 实现(Implementation)

- `csrc/moe/moe_v2.hpp` — `MOE_V2<WeightTraits, ActivationType>`:配置、权重快照(见下方修复)、`forward_many`、逐 token 编排。
- `csrc/moe/moe_v2_bf16.hpp` / `moe_v2_fp8.hpp` — BF16 / FP8 weight traits。
- `csrc/moe/moe_v2_packed4.hpp` — 共享 E2M1 打包 int4 内核,由 `Packed4WeightTraitsBase<LUT, Tag, E8M0>` 参数化(WNA16 / MXFP4(E8M0=true) / NVFP4)。
- `csrc/kernels/bf16_gemm.hpp`、`bf16/utils` — gemm 与激活(bf16 辅助)。
- `csrc/python_binding/binding.cpp` — pybind 模块,导出 `MOE_BF16/FP16/FP8/FP16/WNA16/MXFP4/NVFP4[_FP16]` 及 `forward_many`、`cpu_decode`、`cpu_prefill`。
- `scripts/build_variants.sh` — 从同一源码经 `-DXIAOTU_MOE_MODULE_NAME` 编译 5 个 ISA 变体(`scalar`/`avx2`/`avx512_base`/`avx512_vnni`/`avx512_bf16`)。
- `xiaotu_moe/loader.py` — `_dynamic_loader` 在导入时选最佳 ISA。
- `integration/` — 补丁与干净的 `routed_experts.swap_xiaotu.clean.py`,把 fork 里 `lk_moe.*` 换成 `xiaotu_moe.*`。

### 权重布局(与 fork 喂入对齐)

每个专家块(已用真实 checkpoint 核实,全 checkpoint 的 int8 / e8m0 格式):
- w13 `[E][2I][H/2]`、w2 `[E][H][I/2]` — 沿 K 打包 int4(低 nibble=偶 k、高 nibble=奇 k)。
- 逐块 scale `[E][N/groupN][K/groupK]`:MXFP4 / NVFP4 时把原始 `fp8_e8m0` 单字节指数 scale 解读为 `2^(byte−127)`;groupN=1、groupK=32。
- NVFP4 另有每专家 global scale。

## 5. 阶段状态(Phase Status)

| Phase | 范围 | 状态 |
|---|---|---|
| 1 | BF16 CPU 闭环 | ✅ |
| 2 | FP8 / WNA16 / MXFP4 CPU 内核 | ✅ 已验证(A/B) |
| 3a | NVFP4(策略 A) | ✅ 已验证 |
| 3b | 纯 CPU + 复用 fork GPU MoE | ✅ 已决策并记录 |
| 4 | 多 .so 变体 + 加载器 + 集成 | ✅ 端到端验证 |

## 6. 首个端到端 bug:权重悬垂指针 → SIGSEGV(根因与修复)

在真实 fork + 真实 checkpoint 集成时,EngineCore 在首次 CPU 前向 segfault。崩溃指令 `vmovd (%r12,%rcx,1),%xmm0`(MXFP4 `gate_up` 沿 W 读);fault addr = `w13_ + 254×8MB`(layer-0 专家 254)。

证据链:
- **非竞态**:单线程 pre-scan 读同一地址照样崩。
- **`/proc/self/maps`**:layer-0 权重区几乎全部落入约 2.25GB 的 `---p`(PROT_NONE)匿名保护区。
- **对照**:还原 pristine fork(`import lk_moe`)+ 同一 checkpoint + 真实 prompt 产出正确结果(" Paris. The capital of Spain is Madrid"),证明 lk_moe **在构造时就把权重快照进自身缓冲**。

**根因:** vLLM 的 `clean_weights_after_loading` 会在加载后 `delattr(self, 'w13_weight')`;torch 释放原 param 张量、页被回收或置 PROT_NONE。只存 `data_ptr()` 内核指针的重实现,会在前向时读到已释放内存 → SIGSEGV。

**修复:** `MOE_V2` 构造器改为**把 w13/w2/scale 拷贝进引擎独占的 `std::unique_ptr<uint8_t[]>` 缓冲**,成员指针指向副本——与闭源引擎的所有权模型一致。代价:每层构造多 ~2.7s(全模型加载 ~127s),原张量仍存活时瞬时多占 ~137GB(机器 1.3TB,测试时空闲 ~1.18TB)。

## 7. 验证(Verification)

- 真实 checkpoint(DeepSeek-V4-Flash-0731)+ fresh fork 环境 + 干净构建:`Application startup complete`,发真实 prompt → **HTTP 200,输出与 lk_moe 逐字节一致**,0 segfault。长生成(28-token 俳句)亦稳定。
- cp310 + cp312 × 5 ISA 变体全部重建成功;模块导入并导出全部 12 个 `MOE_*` 类;`load()` 存在。
- 更早的纯 CPU numpy golden A/B:MXFP4 max rel ~3.4e-4、NVFP4 ~8.7e-6,远优于 lk_moe 的 ~22% 尾部。

## 8. 交付物 / 产物(Deliverables)

- `csrc/moe/moe_v2.hpp` — 权重快照修复(核心)
- `csrc/moe/moe_v2_packed4.hpp` — `kE8M0` trait
- `integration/routed_experts.swap_xiaotu.clean.py` — 干净模块替换文件(已装入 fresh env site-packages)
- `build/*.so` — 12 变体(6 ISA × cp310/cp312)
- 文档:`LK_MOE_REIMPLEMENTATION_PLAN.md`、`LK_MOE_INVESTIGATION_REPORT.md`

## 9. 已知差距与下一步

- 仅在 `Lvllmds4-x` 上实测;`Lvllm`(平行线)按 ABI 判定等价(见调查 §1.5),但未单独起服务实测。
- A6000 / 无 FP8 tensor-core 回退线以及真实 WNA16 模型未在集成环境实测。
- 下面两个优化**已实现并实测**(见 §10)。基准机被一个无关的常驻 vLLM 服务器(~90 核)持续吃满,时序数据仅作量级参考、非干净的 Roofline 数据。

## 10. 性能优化(实测)

### 10.1 专家归组 — 已实现,**无提速,诚实的负面结果**

`forward_many` 重写为按活跃专家各走一次,而不是按 token×rank:
1. `per_expert[e]` 为每个活跃专家保存 assignment 下标列表(跳过 `w==0`)。
2. **自适应派发**:`n_active×F ≤ NASS`(能装进 scratch,F 默认 8,可用 `XIAOTU_MOE_GROUP_FACTOR` 覆盖)走 grouped 路径,否则回退逐 token——两条路径位级可比,便于 A/B。
3. grouped 路径为 **3 阶段**(消除跨专家 CAS 竞争):
   - Phase1(按 job 并行):gate/up + SiLU → `act_scratch_`
   - Phase2(按 job 并行):bf16 + down → `down_scratch_`(不写共享输出)
   - Phase3(按 token 并行):`out[t] = Σ_r w·down_scratch_[ai]`(每 token 一线程,按 rank 序累加,无竞争)

**正确性**:grouped 与 per-token 参照**逐位一致**(max abs 0、max rel 0);与 numpy golden 相比仍是通常量化误差(MXFP4 在 |ref|>0.05 上 max rel ~8.7e-4)。

**实测(机器受持续外部负载影响,仅量级参考):**

| batch | per-token(diverse 路由)ms/layer | grouped(集中路由)ms/layer |
|---|---|---|
| 1  | 76 | 74 |
| 8  | 76 | 74 |
| 16 | 76 | 110 |
| 32 | 152 | 213 |
| 64 | 303 | 424-430 |

归组**没有降低延迟**;大 batch 下反而明显变慢。**根因**:每次 gate_up/down 仍按 assignment 读各自 12MB 专家块(块远大于 L2,跨 assignment 不缓存)——所以仅归组并没有实现"每块只读一次"。真正"每块只读一次"需要**分块/批量 GEMM + register blocking**,超出本阶段,留作 future work。此诚实负面结论已写入仓库,避免重复踩坑。

### 10.2 Backend_NUMA 等价物 — 持久 NUMA 线程池(已实现,已完成修复)

`csrc/moe/numa_pool.hpp` 提供 `NumaWorkPool`:持久 worker 池,替换每次调用即建即拆的 `ThreadPool`。
- 持久线程(构造时创建一次,跨调用复用)——去掉每调用建线程的开销。
- NUMA 亲和:从 `/proc/cpuinfo` + `/sys/devices/system/node` 探测拓扑,每个 worker 用 `sched_setaffinity` 钉到不同物理核、跨 NUMA node 轮转;内存交织用 raw `mbind` syscall——**不依赖 libnuma**。
- 动态调度:共享原子索引 `counter_.fetch_add(1)`(隐式工作窃取);`parallel_for(n,fn)` 用类代数(generation)完成屏障。
- 线程数默认=hardware_concurrency,可用 `XIAOTU_MOE_THREADS` 覆盖。

**完成屏障 bug(发现并已修)**:原共享 `done_this_gen_`+`completion_gen_` 计数器存在跨代 reset 竞态,会让 `parallel_for` 在 worker 仍在跑时就返回(压力下 ~80% 崩溃;ASan 证明 worker 在 caller 的 numpy 数组被释放后仍读它)。**修复**:逐 worker 完成记录(`worker_gen_[w]`,`std::atomic<uint64_t>[]`);每个 worker 只写自己的槽位,caller 等每个槽位都等于当前 generation 才返回。无共享累计计数器 reset → 无跨代歧义,可证明不早退。20/20 压力运行 0 崩溃、无死锁。

> **关于"假挂起"的说明**:本机一个无关的常驻 vLLM 服务器(端口 8070,~90 核)把机器 load 顶到 >130;该负载下 pool 的睡眠 worker 偶发被 `cv_` 唤醒要 ~5s,会自恢复,并**非死锁**。在本机测时序/pool 必须计入后台负载;纯 `pthread_barrier`(热驻留线程)实测 ~13µs。
