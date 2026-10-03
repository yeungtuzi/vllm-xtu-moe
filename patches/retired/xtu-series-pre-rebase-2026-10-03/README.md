# rebase 前的旧系列(2026-10-03 存档)

**基线**:vLLM `35d6fb3187`(旧 origin/main);**17 条**。
在服务的树当时 = 该基线 + 这 17 条(逐字节已验证)。

**为什么存档**:rebase 到新上游 `bc21cba967` 后,本目录的 0015/0016(decoder-side SWA
bounded replay)与 `patches/upstream/carry-ced-decoder-replay.patch` **一并退役** ——
上游已自带完整实现(`decoder_replay_layers.py` + `ReplayAttnMetadata`),我们的旧版用的是
已被取代的 `replay_batch` 接口,接不上 ⇒ 表现为 CED 开关无效果。

**回滚**(如需回到旧状态):
```bash
cd /home/user/lvllm/process_data/ref/repos/vllm-mainline
git reset --hard 35d6fb3187
# 再按本目录的 series 逐条 git apply
```
