# GPU/CPU Mixed 模式设计(把 routed experts 放 CPU,注意力留 GPU)

日期:2026-09-08
状态:设计稿(实现待 GLM-5.3-Flash 落地后按此走)
前置:主线 vLLM commit 6c73b08 + torch 2.13.0+cu130(A100 SM80,AMD EPYC 9654 无 AMX)。
作者已在 GitHub fork 主线,本设计对应的改动测试完毕后可推 fork / 提 PR。

## 0. 出发点与原则
- 目标模型族:DS-V4-Flash / GLM-5.3-Flash / qwen3.8-flash 这类**新一代中等规模 MoE**。
- 要解决的问题:大/中 MoE 模型的专家权重(几百 GB to 几十 GB)在单卡(A100 40GB)SMM80 上
  materialize 会 OOM(已实测 DS-V4 138GB 全-GPU 与 offload 均在构造期 OOM)。
- 原则(用户明确):**任何改动应是"合理扩充/修复主线功能、能提 PR",而非专用 trick**。
- 结论:把"Hybrid GPU(注意力)+ CPU(MoE)"做成 **vLLM 的一等公民设备模式**,而不是插件里的 hack。
  参考现状:主线有 `--device gpu|cpu` 两种;**Mixed = GPU 平台跑注意力 + 专家权重/计算放 CPU**。

## 1. 用户接口(新增一处,优雅最小)
- 环境变量 `VLLM_EXPERTS_LOAD_DEVICE`(默认 `gpu`;`cpu` 表示专家放 CPU)
- CLI 参数 `--experts-load-device {gpu,cpu,cuda}`(映射到同一配置字段,二者等价,CLI 优先)
- 注意:`--device` 仍是 `gpu|cpu`(整机);`--experts-load-device` 只管"专家那部分",二者正交。
  - `--device gpu --experts-load-device cpu` ⇨ **GPU/CPU Mixed 模式**(本特性核心)
  - `--device gpu --experts-load-device gpu` ⇨ 现状(全 GPU)
  - `--device cpu` ⇨ 现状(全 CPU)
- 配置存放:`VllmConfig.device_config` 增加字段 `experts_load_device`(或放 kernel_config)。

## 2. 为什么"专家放 CPU"能解决架构/架构问题(M5? 但 M2 已验证)
实测(见 results.txt M2):DS-V4 全 GPU 与 `--offload-backend prefetch` 都在**模型构造期**
GPU ~41GB OOM —— offload 只能搬"已构造"的权重,阻止不了构造期分配。=> 必须让每层 ffn 从
**构造起**就持 CPU 权重。Mixed 模式 = 在 `RoutedExperts.__init__` 建专家参数时直接放 CPU。

## 3. 参数放 CPU 的干净机制(非 trick)
- 主线 `RoutedExperts.__init__`(routed_experts.py:177)调
  `self.quant_method.create_weights(layer=self, **moe_quant_params)` 创建专家参数;
  `Mxfp4MoEMethod.create_weights` / `Fp8MoEMethod.create_weights` 里的 `torch.zeros`
  **没给 device** → 遵循当前默认设备(GPU 平台下是 cuda)→ 大 MoE 就 OOM。
- 干净解法:当 `experts_load_device == "cpu"` 时,把这一处 `create_weights` 调用包进
  `with torch.device("cpu"):`。这是 torch 官方的**作用域式默认设备**机制(实测 `empty→cpu`),
  只影响参数分配,不会把路由表/attention(显式 cuda 或由 manager 建)搬到 CPU。
- 好处:专家参数从创建起就在 CPU,`weight_loader` 拷贝进参数时原地保留 CPU(不 OOM),
  与既有 offload/加载路径零冲突。
- 在主线实现:给 `RoutedExperts.__init__` 加一个 if,或在 `create_weights` 前套 context。PR 里
  把它做成正式 API(带注释、测试),而非魔法。

## 4. 计算(apply)走 CPU
- Mixed 模式下的专家 `apply` 不能再走 GPU kernel(参数已在 CPU)。
- 复用主线已有的 **CPU experts**:`CPUExpertsMxfp4` / `CPUExpertsFp8`(monolithic)本就持 CPU 权重、
  CPU 上跑 grouped-gemm;但它们 `_supports_current_device()` 只认 `is_cpu()`(纯 CPU vLLM)。
- 本特性把 CPU experts 的 `_supports_current_device()` 放宽:在**GPU 平台 + experts_load_device=cpu**
  时也返回 True;且把 `create_weights` 关到 CPU。路径:
  - 模型走 `FusedMoEFactory`;当 mixed 时用 `experts_cls=CPUExpertsMxfp4(·)`
    (monolithic) 或提供 CPU apply 的 `routed_experts_cls`。
  - **xiaotu 引擎作为 CPU experts 的实现后端**:本项目 vllm-xtu-moe 的 `XiaotuCPUExperts`
    (AVX512-VNNI,无 AMX)实现同一契约,替代 AMX-only 的 `CPUExpertsMxfp4`(AMD 9654 用得上)。

## 5. 覆盖范围(诚实边界)
- **主流 MoE 模型**(走 `FusedMoEFactory` 默认 `RoutedExperts`)全覆盖:GLM-5.x、Mixtral、
  Qwen-MoE、DeepSeek-V3 等。← GLM-5.3-Flash 是首选验证(见 §6)。
- **DS-V4-Flash 除外**:它的 MoE 走专属 `DeepseekV4MegaMoEExperts`(非 RoutedExperts),
  需要单独适配(本项目已有 `CpuXiaotuMoE`,保持另一个小实现)。
- 纯稠密模型无 MoE,不适用。

## 6. GLM-5.3-Flash 作为首个验证对象
- fp8 e4m3 → A100 SM80 的 GPU/CPU 都认(不是 nvfp4,不会像 DS-V4 那样 SM80 拒载)。
- 已核实 glm5next/nvidia/model.py:225 用 `FusedMoEFactory` → 走默认 `RoutedExperts` → Mixed
  模式/CPU experts 可直接覆盖。
- 下载中:ModelScope `zai-org/GLM-5.3-Flash`(用户后台)。

## 7. 落地顺序(每步本地测实再推 PR)
1. **主线 config**:加 `experts_load_device` 字段 + CLI/env 解析(可独立先合)。
2. **RoutedExperts**:在 `create_weights` 外套 `with torch.device(device)`(CPU 分支)。
   冒烟:构造大 MoE 不 OOM,参数 device=cpu,weight_loader 原地加载。
3. **apply**:mixed 时选 CPU experts(monolithic)或 CPU `routed_experts_cls`;
   数值对查(与 GPU 版 topk 输出比较)。
4. **性能**:GLM E2E TTFT/吞吐;对比纯 GPU;验证"批 ≥64"可达 >100 tok/s
   (engine 实测 110-167 tok/s,见 results.txt M1)。
5. PR:说明动机/测试命令/结果/数值对比/AI 协助声明。

## 8. 参考
- `vllm/model_executor/layers/fused_moe/routed_experts.py`(__init__:177 调 create_weights;
  PluggableLayer.register("routed_experts");无自带 apply,靠 quant_method.apply)。
- `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`
  (Mxfp4MoEMethod / Fp8MoEMethod 的 create_weights,参数=引擎契约形状)。
- `vllm/model_executor/layers/fused_moe/experts/cpu_moe.py`(CPUExpertsMxfp4=785 apply,
  AMX-only `_supports_current_device`)。
- `vllm/model_executor/layers/fused_moe/layer.py:88`(FusedMoEFactory:
  routed_experts_cls 默认= RoutedExperts)。
- `vllm/model_executor/layers/fused_moe/modular_kernel.py:481,781,982`
  (FusedMoEExperts / Modular / Monolithic 契约;`_supports_current_device` 是 abstract)。

## 9. SM80 移植实据(Lvllmds4-x vs Lvllm,为 SM80 上 hyperconnection 的现成做法)
已人工对比两套同作者 fork(Lvllm=SM120 基线,Lvllmds4-x=SM80 生产),核心做法:
- 加 `_use_tf32_hc_prenorm_gemm()`:SM120 或 SM80(或 deep_gemm 支持时)都返回 True →
  HC pre-norm GEMM 走"可移植 TF32 tilelang路径",并用 `_use_sm12x_mqa_fallback()`
  (family(120) or (is_cuda and family(80)))把 DeepGEMM-only 的 HC/MQA GEMM 全部路由到
  `sm12x_deep_gemm_fallbacks`:3D 分批用 `tf32_hc_prenorm_gemm_triton`(Triton),简单 2D 用
  `_tf32_hc_prenorm_gemm_torch`(纯 torch:`out=x.float()@fn.T;sqrsum=x.float().square().sum(-1)`)。
- **删除首层 broadcast 变体**(整体拿掉 `mhc_pre_broadcast_tilelang`),让首层也走常规
  `mhc_pre_tilelang` → 统一、少一个 DeepGEMM 入口。
- 印证:主线 dev 6c73b08 首层 broadcast 漏了该路由(DeepGEMM SM80 拒载)是我修过的那个 bug;
  Lvllmds4-x 是更彻底地把所有 HC/MQA DeepGEMM 点都 fallback 到 Triton/torch。
- 完整逐文件对比 + 网络调研报告:docs/sm120_to_sm80_port_report.md(子代理产出进行中)。

## 10. 既有轮子(纪律 #1 广搜结果):vLLM PR #37190 "MoE expert CPU offloading"
**结论:先复用上游的 WeightProvider 抽象,别自己重造卸载编排。**
- **PR #37190**(open,**未合并**;不在我们这版 6c73b08):`--moe-expert-cache-size N` +
  `--moe-expert-cache-split {token,expert}`;新增
  `vllm/model_executor/layers/fused_moe/expert_weight_provider.py`(`CachedWeightProvider`);
  改 `routed_experts.py` / `moe_runner.py` / `fused_moe_method_base.py` / `fp8.py` /
  `unquantized_fused_moe_method.py`。
- **机制(读 diff 源码)**:全部专家权重在 **CPU pinned 内存**,GPU 只留 capacity 个 slot 的 scratch
  buffer;LFRU(freq/age)淘汰;**未命中 → H2D 拷进 slot → `kernel.apply()` 在 GPU 上算**。
  即"**CPU 存 + GPU 算**",**不是**"CPU 上 AVX/AMX 计算"。
- **量化支持**:首版仅 fp8 + unquantized(`supports_expert_lru_cache()`);**不含 mxfp4/fp4**。
  限制:不兼容 EP>1/DP/SP;缓存 ≥ top_k。
- **可复用点**:其注释明确"kernel 不知道也不关心权重从哪来"——`WeightProvider` 是**可插拔接缝**。
- **与本设计的关系**:PR 的"CPU 存+GPU 算"每次未命中要走 PCIe(26GB/s)搬专家权重;本设计的
  xiaotu 引擎是"**CPU 算**",权重留 CPU、只搬 tiny 激活(CPU DDR5 ~600GB/s),实测带批 110-167 tok/s
  (results.txt)。=> 二者互补。**推荐把 xiaotu 实现为一种 CPU-compute WeightProvider**,插进上游接缝;
  GPU-cache provider(PR)留给"热专家能装进缓存"的场景。DS-V4-Flash(fp4)不在 PR 首版范围,且
  GPU fp4 内核在 SM80 有问题,更需要我们的 CPU-compute 路径。

## 修订记录(纪律 #8)
- **2026-09-08e(本版)** — §4 计算半边**接通(未 E2E)**:主线 oracle/mxfp4.py CPU 回退条件
  放宽为 `is_cpu() or VLLM_EXPERTS_LOAD_DEVICE=="cpu"`;插件新增
  `vllm_xiaotu_moe/mixed_experts.py`(`XiaotuCPUExperts(CPUExpertsMxfp4)`,覆盖设备门控/
  权重后处理/apply,计算调 xiaotu 引擎)+ `register_mixed_cpu_backend()`(混合模式下把
  CPU MXFP4 后端换成我们的类,`__init__` 加载时调用)。实测:=cpu→oracle CPU 后端=
  ['XiaotuCPUExperts']、_supports_current_device=True(AMD 无 AMX);=gpu→no-op。
  **E2E(真实 forward)未验证**,apply 为 scaffold,下一步在可跑模型上验。
- **2026-09-08d(本版)** — §3 的机制**已落地并实测通过**:主线 `envs.py` 新增
  `VLLM_EXPERTS_LOAD_DEVICE`(gpu|cpu),`RoutedExperts.__init__` 在 cpu 时把
  `quant_method.create_weights` 包进 `with torch.device("cpu")`。实测(scripts/
  mixed_load_device_test.py,GPU2,模拟 GPU 默认设备):=gpu→专家参数在 cuda:0(PASS);
  =cpu→在 cpu(PASS)。配置+权重驻留半边完成,可单独 PR;计算半边(apply 走 CPU)待做。
- **2026-09-08c(本版)** — §10 新增纪律#1 广搜结果:vLLM PR #37190(open/未合并)实现
  `--moe-expert-cache-size` + `CachedWeightProvider`(CPU pinned 存 + GPU 算 + LFRU + H2D)。
  明确其机制非"CPU 计算";推荐复用其 WeightProvider 接缝,把 xiaotu 做成 CPU-compute provider。
- **2026-09-08b(本版)** — §9 补充 SM80 移植实据(人工对比 Lvllm(SM120) vs Lvllmds4-x(SM80)):
  `_use_tf32_hc_prenorm_gemm()`(SM120|SM80→True)+ `_use_sm12x_mqa_fallback()`(family120|
  cuda&family80)→ DeepGEMM-only 的 HC/MQA GEMM 全部 fallback 到可移植 Triton/torch;
  删除首层 broadcast 特例、走统一 mhc_pre_tilelang。这与我对主线作的修复一致且更完整。
- **2026-09-08a(本版)** — 新建。把 "Hybrid GPU-attention + CPU-MoE" 定为 vLLM 一等公民
  "GPU/CPU Mixed" 模式(env `VLLM_EXPERTS_LOAD_DEVICE` + CLI `--experts-load-device`),
  参数放 CPU 用 torch 作用域默认设备(非 trick),apply 走主线 CPU experts/或 xiaotu 引擎,
  GLM-5.3-Flash(fp8, A100-OK, 走 FusedMoEFactory)为首个验证对象。覆盖主流通用 MoE 模型,
  DS-V4-Flash 专属除外。作者可 PR。
