# 已退役的 patch

## carry-ced-decoder-replay.patch(2026-10-03 退役)

原因:**上游 `origin/main` 已自带完整实现**(`vllm/models/deepseek_v41/decoder_replay_layers.py` +
`nvidia/model_state.py` 的 `ReplayAttnMetadata` + `nvidia/model.py` 集成)。

我们那份是**旧版 #56752** 的移植,用的是 `DecoderReplayLayers.replay_batch` 接口;
上游后来改成 `ReplayAttnMetadata.replay_start`,两边**接不上** ⇒ 我们的模块成了死代码
(全树无人给 `replay_batch` 赋值),表现为 **CED 开关对性能零影响**(实测 CED=0 527.8 vs
CED=1 527.9 tok/s)。rebase 到新上游时 0015/0016 与该文件**一并退役**。
