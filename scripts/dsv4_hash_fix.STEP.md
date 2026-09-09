# DS-V4 hash 层路由修复 — STEP 日志(增量)

Date: 2025-09-09 (session, agent)
Checkout: `/home/user/lvllm/process_data/ref/repos/vllm-mainline` + 插件 `/home/user/lvllm/vllm-xiaotu-moe`
Env: `conda activate vllm-xiaotu-moe`, 只动 GPU2(CUDA_VISIBLE_DEVICES=2),不碰 prod/fork。

## 0. 结论速览(先看这个)
- **hash 层 (0/1/2) 的 `topk_ids` 其实已经是"正确的非 -1、且随 token 变化"**——真实前向里用
  `tid2eid[token_id]` 查表没问题(表已从 checkpoint 正确加载,值域 0..255)。
- **真正让整体输出全 0 的根因 = 每层 MoE 的 `hidden_states` 输入(`hin`)在真实生成时是 0.000e+00**,
  尤其是 **layer0 的 ffn 输入就已经是 0**。输入是 0,无论路由对错,引擎 output = 0 → 逐层 0 →
  lm_head(0) argmax = token 0 → 所有旧日志 token_ids 全 0(见 dsv4_real/out、dsv4_dbg/out、dsv4_topk.out)。
- 因此"修 hash 路由"**不会** 让输出非 0;needs 修的是"为什么 layer0 之前的 pre-FFN hidden = 0"。
- 这不是 MoE/hash 的问题,是 **embedding→attention(pre-FFN)= 0** 的问题,疑似 SM80 上
  MHC(hyper-computing tilelang)注意力路径对某个形状输出 0。

## 1. 已确认事实(证据)
- checkpoint 含 `layers.{0,1,2}.ffn.gate.tid2eid`,int64,shape [129280,6],min=0 max=255(合法专家 id)。
- 插件 `CpuXiaotuMoE.forward` 的 `fused_topk_bias(...)` 调用与主线 mega 版逐参一致:
  `indices_type=hash_indices_dtype=int64`、`input_tokens=input_ids`、`hash_indices_table=gate.tid2eid`。无参数/顺序差异。
- fork `Lvllmds4-x` 的 `fused_topk_bias_router.py` 对 hash 用**同一个** `ops.topk_hash_softplus_sqrt`
  内核;fork 无独立 `dsv4_topk`(那是主线新增的 triton 优化)。=> hash 路由两者同内核,无 SM80 差异。
- hash 内核 `topk_softplus_sqrt_kernels.cu`:`dsv4HashTopkSoftplusSqrt` 读 `tid2eid[token_id*6+lane]`,
  仅当 `is_pad_row` 时写 `-1`/`0`。`_get_padding_mask` 在 `VLLM_MOE_SKIP_PADDING=1`(默认)时提供 `is_padding`。
- 实测(重跑,XIAOTU_DEBUG_L1=1,明文进度):唯一一次 hin≠0 是 8192 token 的 warmup/profile pass
  (hash 层 -1/0 = 该 pass 全为 padding 的正确行为;非 hash 层经 dsv4_topk 忽略 padding → 输出非 0)。
  之后真实 5-token prefill 与各 decode step:layer0 hin=0.000e+00,hash topk **非 -1、逐 token 不同**
  (layer0 = [[147,78,30,248,217,179],[188,165,...],...]) → 路由本身工作正常。
  但 hin=0 → 引擎输出 0 → 整体 token 全 0。

## 2. 位置定位(进行中)
- layer0 ffn 输入 = `x(embedding) → mhc_pre_broadcast_tilelang → self.attn → mhc_fused_post_pre_tilelang → x`(见 nvidia/model.py 1186-1270)。
- 8192-token warmup pass hin=0.177 ≠ 真实 5-token prefill hin=0.0 ⇒ 疑似按 M/形状分叉:
  `x.dim()==2 → mhc_pre_broadcast_tilelang`(broadcast)vs 其它。怀疑 SM80 的 tilelang MHC 在
  broadcast/小 M 下产出 0(broadcast 依赖 `hc_attn_fn_broadcast`)。
- 【待确认】embedding 本身非 0(hin 反映的是 pre-FFN,非 embedding);下一步:在 decoder 里
  mhc_pre 前/attention 后各打一点,确认 0 出现在 o_proj 之前还是之后。

## 3. 试过/打算试的修法
- [x] 确认路由对齐主线 mega(无需改,已对齐)。
- [x] 验证引擎本身正常(isolated cpu_decode_test: real_layer1 weights + 随机输入 → out 全非 0)。
- [ ] `VLLM_MOE_SKIP_PADDING=0`(关 padding mask)一测——预计无用,因真实 prefill hash 已非 -1。
- [ ] **若确认 SM80 MHC/attention 输出 0:** 需在注意力/MHC(mhc_pre/mhc_post/attention)更靠前处
  路由/修 fallback,而不是 MoE。注意 fork 是全局禁 DeepGEMM(has_deep_gemm→False)+ xpu 路径,不可照搬。

## 4. 下一步
定位 0 出现的位置(pre-FFN hidden),看是否 SM80 tilelang MHC/attention 分支所致,再决定改哪。
