# 已知问题:LMCache 会导致 **长 prompt 崩溃** 与 **输出退化**

> ⚠️ **TL;DR —— 结论先给**:
> **本项目默认关闭 LMCache**(`LMCACHE=0`,且不启动 `lmcache_server`)。
> 在我们这台机器上,LMCache 是 **long prefill 恶性崩溃** 的**唯一变量**,
> 并且与**输出退化(重复、意图漂移、写错路径)**高度相关。
> **在上游修复前,请勿在生产开启。**

---

## 1. 我们实测到的症状(2026-10-05,本机,DeepSeek-V4.1-Flash,TP=2,A100×2)

### 1.1 长 prompt 直接打挂引擎(恶性 ✗)

崩溃栈(两次独立复现,配置相同):

```
[LMCache ERROR] Something went wrong when processing the retrieve request for request_id=chatcmpl-...
(EngineCore) RuntimeError: A KV connector reported block-level load failures (invalid_block_ids)
             which is not supported for mode...
   @ vllm/v1/core/sched/scheduler.py:2019  _handle_invalid_blocks
   @ vllm/v1/core/sched/scheduler.py:3310  raise RuntimeError
⇒ APIServer: EngineDeadError  ⇒ 整个服务退出
```

⇒ **这不是"拒绝请求",而是【整进程崩】**:一次大请求就能让 **所有** 会话断掉。

### 1.2 对照实验(唯一变量 = LMCache ✓)

| 配置 | 冒烟 | 7.3 万 | **20 万** | **78 万** | `device=cuda`(GPU 预填)| 危险关键词 |
|---|---|---|---|---|---|---|
| SPEC=0 + **LMCACHE=1** | 200 | — | ⭐ **崩(两次都崩)** | — | **0** ✗ | **1 / 2** ✗ |
| SPEC=0 + **LMCACHE=0** | 200 ✓ | 200 ✓ | ⭐ **200(209 s)** ✓ | ⭐ **健康推进** ✓ | ⭐ **686** ✓ | ⭐ **0** ✓ |

⇒ ⭐ **去掉 LMCache 后**:长请求成功、**GPU 预填从 0 次恢复到 686 次**、零危险关键词。

### 1.3 输出退化(与上游报告一致,我们观测到但未做受控实验)

在开启 LMCache 的时段,我们观察到 agent 会话出现:
**输出重复**、**意图漂移**(反复说"让我开始"却不发工具调用)、**把项目路径写错**
(`vllm-xiaotuo-moe` / `lvmlm` / `xiaomju-moe` 等 → 见 `dev-docs/BUG_REGISTRY.md` A14–A38)。
⚠️ **这属于"与上游报告一致"的关联,不是我们证明的因果** —— 详见 §4。

---

## 2. 上游已知问题(公开 issue,建议直接跟踪)

| 出处 | 关键内容 |
|---|---|
| [LMCache **#1064**](https://github.com/LMCache/LMCache/issues/1064) | *GPU buffer memory reuse bug causing CacheBlend **incorrect model outputs** on subsequent prompts* |
| [LMCache **#4674**](https://github.com/LMCache/LMCache/issues/4674) | *LMCacheMP retrieve still **near-corrupts on multi-session prefix hits*** |
| [LMCache **#4984**](https://github.com/LMCache/LMCache/issues/4984) | *LMCacheMPConnector + **MTP**: **deterministic corruption** on multi-step retrieves(**#4253 修复后仍存在**) * |
| [LMCache **#1866**](https://github.com/LMCache/LMCache/issues/1866) | *response is wrong due to async KV loading using **wrong memory object*** |
| [vLLM **PR #45146**](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/45146/overview) | *[BugFix] Reset num_output_placeholders on **KV load failure** recomputation* |
| [arXiv 2609.38706](https://export.arxiv.org/pdf/2609.38706) | *Preserving Provenance in Shared KV Caches for LLM Serving*(审计 connector 全路径)|

⇒ 归纳机制:**GPU 缓冲区复用 / async KV 载入配错 memory object ⇒ 取回的 KV 与当前请求不匹配 ⇒
模型读到别的上下文 ⇒ 输出重复/漂移,或调度器直接判定 `invalid_block_ids` 而崩**。

---

## 3. 怎么关闭(本项目已默认关闭 ✓)

```bash
# 引擎侧:不连接、不加载 LMCache 连接器
LMCACHE=0 bash scripts/bringup_prod_8070.sh        # 生产脚本已把默认值改为 0 ✓

# 服务端:不启动(省 GPU 0.85 GiB + 宿主 ~65 GiB RSS)
bash scripts/proc.sh stop lmcache_server
```

**关闭后的代价与收益**

| | 影响 |
|---|---|
| ⚠️ 代价 | 失去**跨会话/跨进程的 KV 复用**(长会话首 token 可能变慢)|
| ✅ 收益 | **long prefill 不再崩**;⭐ **GPU 预填恢复**(`device=cuda` 0 → 数百次);GPU 省 **0.85 GiB**;宿主省 **~65 GiB RSS** |

⚠️ **注意**:关闭后 `5555`(LMCache MP)与 `8080`(LMCache HTTP)两个端口**预期为 DOWN**,
这不是故障 —— 见本仓启动脚本的端口自检说明。

---

## 4. 非声明(我们**没有**证明的事)

* ❌ **没有**证明"LMCache 导致退化"的**因果** —— 我们只有**长 prompt 崩溃**的对照实验(§1.2 ✓),
  退化部分只有**症状观察 + 上游报告吻合**。
* ❌ **没有**做多轮重复(§1.2 每个配置的长请求各 1~2 次)⇒ **不能声称已彻底验证**。
* ❌ **没有**定位到 LMCache 内部具体哪个缓冲区配错内存(那是上游的工作)。
* ✅ **已做**:每次改动前都有配置快照,所有结论(含我们自己的错误)都记录在
  `dev-docs/BUG_REGISTRY.md`(A14–A38)。

---

## 5. 回滚(如果你确实需要 LMCache)

1. `scripts/bringup_prod_8070.sh` 里把 `LMCACHE="${LMCACHE:-0}"` 改回 `-1`;
2. 启动服务端:`bash scripts/proc.sh spawn lmcache_server env "${LMCACHE_ENV[@]}" bash scripts/serve_lmcache.sh`(见启动脚本 §LMCache);
3. 重启 vLLM;
4. ⚠️ 但请预期 **长 prompt 崩溃** 与 **输出退化** 会回来 —— 除非上游已修复 §2 里的 issue。

**配置快照**(可直接取回改动前的版本):
`dev-docs/vllm_snapshots/bringup_prod_8070.sh.20261005-154430`
