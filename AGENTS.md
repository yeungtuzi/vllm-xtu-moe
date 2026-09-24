# AGENTS.md —— 本仓库对自动化代理的强制纪律

## ⛔ 第 0 条(最高优先级;违反即事故):进程管理

1. **禁止按"名字 / 命令行模式"杀进程**:不得使用 `pkill -f <pat>`、`ps aux | grep <pat> | ... kill`、
   `pgrep -f <pat> | xargs kill` 等。**你的命令行本身包含该模式 ⇒ 必然把自己杀掉**
   (本仓库历史上已发生 40+ 次,每次都造成会话中断与现场丢失)。
2. **启动进程必须走 `scripts/proc.sh spawn <name> <cmd...>`** ⇒ 在 `dev-docs/report/tuning/logs/` 下生成
   `<name>.pid` 与 `<name>.log`:PID 可追溯、**日志持久化**(不随工具调用结束丢失)。
3. **停止进程只能走 `scripts/proc.sh stop <name>`**(内部只读 PID 文件,按进程组停止)。
4. **没有 PID 文件时**:先用**端口派生 PID**(`ss -ltnp | grep ':PORT'`)或
   `nvidia-smi --query-compute-apps=pid`,再用 `scripts/proc.sh adopt <name> <pid>` 登记;
   **禁止**退回名字匹配。
5. **查询**:用 `scripts/proc.sh status <name>`;确需列出进程时,过滤条件必须**避开自己命令行的字面量**。
6. **长驻服务(>10 分钟)**:一律使用**受管后台任务**或 `proc.sh`;不要随手 `setsid ... &`
   (可能被回收,且日志会丢)。
7. **禁止无条件清理 GPU 进程**:`for x in $(nvidia-smi --query-compute-apps=pid ...); do kill -9 $x; done`
   会杀掉**同机共存的其它服务**(LMCache 服务自身就用 CUDA ⇒ 它也在该列表里 ✗;本机已因此把 LMCache
   服务杀了两次 ✗)。**停服务只能停自己 PID 文件里那一个** ✓;确需清理"残留"时,先明确列出允许停的 PID
   并逐个确认归属,再杀 ✓。
8. **任何"批量 kill"前必须自问**:这条命令会不会命中(i)我自己、(ii)同机共存的其它服务?
   只要有一丝可能,就改用 PID 文件 / 端口派生 PID 的精确写法 ✓。


## 其它既有纪律(见 dev-docs/HANDOFF_v0.2.5.md §5)

- 修改 shell 脚本:`bash -n` **不够**(查不出续行链被注释打断),必须加"续行链结构校验"。
- 启动服务后**必须同步读启动日志**确认,不许"发脚本→等结果"。
- 改投机 / 图模式 / RoPE 前先看 `Mean acceptance ratio`;`GPU prefill ACTIVE` 不等于已启用,要看 slack 正负。
- 结论写入台账(`docs/EXPERIMENTS.md`),不要只留在会话里。

9. **"清理残留"也必须用显式 PID 允许列表**:从**日志/PID 文件**里把要停的 PID **逐个抄出来**,
   确认归属后**逐个 kill** ✓;**禁止**任何形式的模式匹配(包括 `case "$cmdline" in *xxx*`,
   `grep`+xargs 等 ✗)。**教训**:本会话曾用 `case` 匹配 `*lmcache*` "清理残留",
   结果**把本该保留的 LMCache 服务端杀了** ✗ —— 模式匹配在清理场景同样会命中错误目标 ✓。

## 其它纪律补充

- **`cmd | tail` 会掩盖退出码** ✗(本会话已踩:编译其实失败 `Cannot find CMake executable`,
  却因管道返回 `tail` 的 0 而被误判为成功 ✗)。**构建/关键命令必须**:
  重定向到日志文件(或 `set -o pipefail`)后再判断退出码 ✓。
- **跨树运行 vLLM 的硬约束**:`PYTHONPATH=<另一棵树>` **只能换 Python 代码,换不了已编译扩展** ✗
  ⇒ 目标树必须**树内有 `.so`**(即就地 `setup.py build_ext --inplace` 过 ✓);
  否则报 `vllm_flash_attn requires the CUDA flash attention extensions` ✗(见 EXPERIMENTS B179 ✓)。

## ⛔ 工作树纪律(用户 2026-09-23 指示)

**任何需要"再开一个临时 worktree"的情况,必须先警告用户,由用户取舍**:
* 立即关闭生产服务腾出主树 ✓,或
* 延后这项开发 ✓
**不得自行另开工作树** ✗ —— 历史教训:分叉出 7 棵树(每棵树各有独立 `.so`、生产又靠 `PYTHONPATH`
指向具体路径 ✗),收敛成本极高 ✓。**默认保持"一棵指定树"** ✓;确需预演时,先在 `/tmp` 下做、
用完即删 ✓,并事先告知用户 ✓。

## ⛔ 外网访问纪律(用户 2026-09-23 第二次提醒)

**访问外网(下载、gh、curl、pip、clone 等)必须显式带上本机代理**,否则会被挡住:
```bash
export HTTPS_PROXY=http://192.168.195.21:8080 HTTP_PROXY=http://192.168.195.21:8080 \
       https_proxy=http://192.168.195.21:8080 http_proxy=http://192.168.195.21:8080
```
* 代理值来自 `~/.bashrc`,但**非交互 shell 不读它** ⇒ **必须在命令里显式 export**
* 典型症状:下载“看似成功”实则被截断/为空、`gh search` 返回空 ⇒ **下载后必须校验大小与 Content-Length 一致**
* 判据:`curl -sI https://...` 应看到 `Proxy-agent: tinyproxy/...`
