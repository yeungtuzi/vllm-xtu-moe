# vllm-xtu-moe: 主线 hybrid 适配器设计(把 DS-V4 的 MoE 层挪 CPU 走 xiaotu)

日期:2026-09-08
状态:设计稿(供评审;实现前先钉清楚,避免在共享机器上反复重型试错)
前置:环境已就绪(torch 2.13.0+cu130 + vLLM 主线 0.1.dev1+g6c73b08de + 全依赖)。
      注入点已实证(见 ref/fork_vs_mainline_plugin_decision.md §11:DS-V4 MoE 用 FusedMoEFactory,
      签名含 routed_experts_cls / experts_cls 选层)。
目标:在**主线 vLLM**(非 fork)上,像 lvllm 那样做混合推理:把 DeepSeek-V4-Flash 的 MoE 层
      放到 CPU 用 xiaotu 引擎算,GPU(A100)专注注意力/其余层,解决 138GB MoE 装不进 2×40=80GB
      的问题,并尝试达到 fork 的 ~100 tok/s decode。

## 0. 为什么是"层级 hybrid"而非"算子级替换"
- 138GB fp4 MoE > 80GB(2×A100)= 不可能全载 GPU。必须把 MoE 从 GPU 挪走。
- 主线 DS-V4 nvidia/model.py **没有原生"层 CPU-offload"开关**;但 MoE 由 FusedMoEFactory
  构造,它接收 `routed_experts_cls`。=> 最干净的注入 = 提供自定义 `RoutedExperts`(或直接按层
  在 DS-V4 model 的 experts 构造点换成我们的 CPU MoE 模块)。
- 与 fork 对齐:fork 用 is_lk_moe_cpu_layer(layer_name) 逐层标记 CPU/GPU;主线侧我们用"层
  构造点换 experts 类"达到同样效果,且不 fork 主线源码(改 我们的 plugin,即 OOT 覆盖/注入)。

## 1. 整体数据流(一层,decode/小 batch 形态)
GPU 注意力层产出的 hidden_states(dev,cuda) →
  ① H2D? 否 —— hidden_states 已在 GPU;要 CPU 算 MoE 需 D2H 到 CPU pinned 缓冲;
  ② xiaotu 引擎 CPU 算该层 MoE(hidden D2H → forward_many(CPU) → 输出 H2D);
  ③ 输出回 GPU,继续下一层注意力。
即 fork 的 cpu_decode 形态:stream-enqueued D2H + host-func(CPU MoE) + H2D,
一次一层、串行(无 overlap,显存省),与 lk/xiaotu fork 一模一样。

## 2. 三个需要落地的部分
A. **引擎接入**:xiaotu 引擎(csrc/xiaotu_moe)需要能在这台 A100+EPYC 9654 上跑 down-to-GPU 的
   MoE。fork 里它是 lk_moe 的 drop-in(loader 直接给 MOE 类),绑定 API 已实证
   (MOE(cfg,w13,w2,w13_scale,w2_scale,w13_global_scale,w2_global_scale).cpu_decode(stream,qlen,
   top_k,hidden,ids,weights,out_gpu) / .cpu_prefill(qlen,top_k,ids,weights,input,output))。
B. **层标记/注入**:在 DS-V4 model 的 experts 构造处(FusedMoEFactory / experts_cls)按层换
   成我们的 hybrid 版本;或提供 OOT plugin 修改选择。参考 fork 的
   is_lk_moe_cpu_layer(layer_name) 语义 → 主线侧我们定义一个"哪些层走 CPU"(默认或按配置)。
C. **权重桥**:主线 DS-V4 层加载后的权重(w13/w2/scale)在 vLLM 的 per-rank 张量中;xiaotu 引擎
   构造时要持 CPU 权重镜像(cpu 布局)。两处数据类型/布局需对齐(MXFP4 / fp8 / awq…)。

## 3. 关键设计决策(先定,再码)
1. **计算内核契约**:主线 `FusedMoEExperts.apply()`(monolithic + modular 两类) vs
   **fork 的 RoutedExperts(forward_modular/forward_monolithic)**。二者都是 vLLM 标准入口,
   我们选与主线 DS-V4 实际路径一致的那个(DS-V4 nvidia model 走的 FusedMoEFactory →
   MoERunner → RoutedExperts.forward_modular/forward_monolithic)。
2. **CPU 路径阈值**:沿用 fork —— 小 batch(<min_batch 如 1024)也许仍 GPU/VNNI;大 prefill/ decode
   走 CPU。但 decode 是逐 token,天然小 batch,仍可能 CPU — 需按 fork 的 max_num_group_batch_size
   语义(成组批量)来确定,否则逐 token D2H/H2D 开销不可接受。
3. **NUMA/线程**:主线 CPU worker 已按 rank 绑 NUMA node(VLLM_CPU_OMP_THREADS_BIND auto);
   xiaotu 引擎在 CPU 算该层时用引擎自带线程池,要尊重/对齐 rank 的 NUMA 分区(不跨区读 → 保持
   325GB/s 目标)。见 ref/fork_vs_mainline_plugin_decision.md §9。
4. **不逐位一致**:CPU vs GPU 量化数值非位一致(既有问题),只对齐机制/吞吐,不做逐位校验。
   => 测试判据用 TTFT/吞吐/显存,不比对输出字节。

## 4. 里程碑(DoD,逐步)
M1: 在主线 env 里能把 xiaotu 引擎 import / 构一层 MOE / 跑一次合成 forward_many(离线,GPU2,
     CPU 布局),确认引擎在本机可算 —— 不动模型。
M2: 把某一层 DS-V4 的 MoE 换成我们的 CPU-RoutedExperts(单层注入),GPU2 离线加载模型 + 单步
     forward,确认:该层走 CPU(xiaotu)、输出形状/设备正确、GPU 注意力仍正常。不要求全模型。
M3: 全部 MoE 层走 CPU(或按配置),GPU2 离线起 serve(8071),发请求验证 TTFT/吞吐,显存稳定、
     不 OOM、阈值切换正确。
M4: 对比 fork ~100 tok/s decode:跑同一 benchmark,记录是否达到/接近。

## 5. 风险与诚实的底线
- 主线 DS-V4 model 的层构造点是否稳定到"可被我们干净注入",需在 M2 实测确认(可能它内部
  MegaMoEExperts 把 experts 封装得更深,注入点比 FusedMoEFactory.routed_experts_cls 更藏)。
  **已实证的层结构(09-08)**:DeepseekV4DecoderLayer(model.py:1096)持有 self.ffn = DeepseekV4MoE(
  vllm_config, prefix=f"{prefix}.ffn") + self.attn;层号可从 prefix("layers.{i}.ffn")解析 →
  **逐层 CPU-offload 的 seam 与 fork 的 is_lk_moe_cpu_layer(layer_name) 语义一致**:按前缀/层号
  决定是否把 self.ffn 换成 CPU(xiaotu)版本;GPU 注意力(attn)不动。
- 做法(不 fork 主线源码,保持主线 pristine):`ModelRegistry.register_model("DeepseekV4ForCausalLM",
  "<our_module>:<class>")` 会**直接覆盖**主线的该 arch 注册(registry.py:1133-1139 明确
  "already registered, and will be overwritten")→ 我们的 DeepseekV4ForCausalLM 子类据此替换
  self.ffn(按 prefix 层号)。即:纯 OOT plugin,主线零改动。此机制已源码实证。
- 逐 token D2H/H2D 开销:decode 小 batch 若没做成组批量,per-call 拷贝开销可能吃掉 gain ——
  必须复用 fork 的"成组批量 + 持久 pinned 缓冲"做法(见 fork _cpu_decode)。
- **主线原生 offload 救不了 DS-V4 on 40GB(已实测,09-08)**:
  --offload-backend prefetch + offload-params 专家权重确实激活 PrefetchOffloader,但**模型构造期
  (initialize_model -> DeepseekV4Model)就在 GPU 分配 ~41GB**,offloader(设计成"已构造权重再搬")
  无法阻止构造期 OOM(某 2GB 连续分配 free 仅 737MB)。=> 要让模型构造成功,必须真把每层
  ffn 换成 CPU(自写 hybrid / _wrap_cpu_ffn),原生 offload 不是捷径。
- 138GB 权重 CPU 内存:本机 1.5TB 内存宽裕(widely),无虞;但加载/桥接耗时。
- 共享机器负载(loadavg 高)+ prod 在 GPU0/1:一切测试只在 GPU2/8071,等窗口,不碰 prod。
- **能否到 100 tok/s 需实测,不承诺。**

## 6. 参考点
- fork 的 integration/routed_experts.swap_xiaotu.clean.py(逐层 forward 路由 + _cpu_decode/_cpu_prefill
  + process_weights_after_loading 建 lk_moe/xiaotu 引擎 + is_lk_moe_cpu_layer 逐层标记)是主模板。
- 主线 FusedMoEFactory(layer.py:88,routed_experts_cls / runner_cls)与 DS-V4 nvidia/model.py
  (experts_cls 选 MegaMoEExperts/FI)定义注入面。
- xiaotu 引擎绑定 API(csrc/python_binding/binding.cpp)定义计算入口。

---
## 修订记录
- **2026-09-08g(本版)** — 【方向 pivot】按用户原则("改动须能提 PR、非 trick")+ 用户已在 GitHub
  fork 主线,把"Hybrid GPU-attention + CPU-MoE"定为主线一等公民 **GPU/CPU Mixed 模式**
  (env `VLLM_EXPERTS_LOAD_DEVICE` / CLI `--experts-load-device`,与 `--device` 正交);
  参数放 CPU 用 torch 作用域默认设备(`with torch.device("cpu")` 包 create_weights,非 trick;
  已实测内部 empty→cpu 且不波及 routing/attention),apply 复用主线 CPU experts 或 xiaotu 引擎。
  完整规格见 docs/gpu_cpu_mixed_mode.md。首个验证对象 GLM-5.3-Flash(fp8,A100-OK,走
  FusedMoEFactory)。主线 MHC SM80 修复已做对(PR-ready)。详见 results.txt 通用化 pivot。
- **2026-09-08f(本版)** — §6 E2E 里程碑 + 新硬伤:GPU2/8071 上 hybrid OOT **完整构造成功、全 43 层
  xiaotu 引擎建好、GPU2 仅 ~10GB(138GB MoE 未 materialize)→ 构造期 OOM 彻底解决**;权重绑定
  逐字节验证(real fp4 + raw e8m0 精确入 CPU 参数)。但一次完整 forward 被卡:主线 DS-V4-Flash 的
  MHC(hyperconnection)硬编码 tilelang→deep_gemm.tf32_hc_prenorm_gemm,而 vendored deep_gemm 只
  实现 sm90/100/120,无 sm80 → A100(SM80)报 "Unsupported architecture"。这是主线模型级限制,
  与 MoE-CPU 插件正交。出路:OOT 换 mhc_*_torch(风险高)、SM80 版 deep_gemm、换 SM90/100 机、或收敛。
  详见 results.txt M2/MHC。
- **2026-09-08e(本版)** — 【决定性】真实 E=256 吞吐实测(scripts/m1_bench_e256.py + e256_time.py):
  读真实 layer-3 全部 256 routed experts 建 xiaotu MOE_MXFP4(gk32 raw e8m0),EPYC 9654 纯 CPU:
  单 token decode 仅 ~2-10 tok/s(延迟受限,且不随核数涨 → 与已观测"48/96/190 核差不多"吻合);
  但带批后 B=64→114, B=128→141, B=256→167 tok/s(MoE-only 天花板)。=> fork 的 ~100 tok/s 是
  "带批吞吐":每层把本批激活专家权重读一次、整批共享。=> 我们的主线插件要达成 ~100 tok/s,关键在
  decode 保持 batch>=~64(max_num_seqs),不是计算内核单流多快。若插件做不到批处理,d 50 上只能个位
  tok/s。engine 可靠性:cpu_prefill 紧循环背靠背偶发 NumaWorkPool use-after-free(SIGSEGV,0xd3
  毒值,OMP=96 过订阅更易触发),单次调用+gc 稳定。落插件避免同一引擎紧循环。详见 results.txt。
- **2026-09-08d(本版)** — §5 补实测:主线原生 PrefetchOffloader(激活成功)救不了 DS-V4 on
  40GB —— 模型构造期即 GPU ~41GB(2GB alloc free 737MB 失败);offloader 只能搬"已构造权重",
  无法阻止构造期 OOM。=> 必须真换每层 ffn 为 CPU(自写 hybrid),原生 offload 非捷径(除非先让
  模型构造成功)。依据:GPU2/8071 两次实测(results.txt M2 / M2-2)。
- **2026-09-08c(本版)** — §5 再补:OOT 覆盖机制实证——ModelRegistry.register_model("DeepseekV4ForCausalLM",
  "<module>:<class>") 直接覆盖主线该 arch(registry.py:1133-1139 明确 overwrite)→ 纯 OOT plugin、
  主线零改动即可用我们子类替换 self.ffn(按 prefix 层号)。
- **2026-09-08b(本版)** — §5 补实证:DeepseekV4DecoderLayer(model.py:1096)持有 self.ffn=DeepseekV4MoE
  + self.attn,层号可从 prefix("layers.{i}.ffn")解析 → 逐层 CPU-offload 的 seam 与 fork 的
  is_lk_moe_cpu_layer 语义一致:按层号把 self.ffn 换 CPU(xiaotu),attn(GPU)不动;保持主线 pristine
  (走 OOT 替换/subclass,不 fork 主线源码)。
- **2026-09-08(本版)** — 新建。给出主线 hybrid 适配器的完整设计:数据流(D2H→CPU forward_many→H2D)、
  三部分(引擎接入/层标记注入/权重桥)、四个关键决策、DoD 里程碑 M1-M4、风险与诚实底线。
  依据:本次会话源码实证(FusedMoEFactory.routed_experts_cls、DS-V4 nvidia/model.py、xiaotu binding)。
