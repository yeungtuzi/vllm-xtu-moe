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

## 其它既有纪律(见 dev-docs/HANDOFF_v0.2.5.md §5)

- 修改 shell 脚本:`bash -n` **不够**(查不出续行链被注释打断),必须加"续行链结构校验"。
- 启动服务后**必须同步读启动日志**确认,不许"发脚本→等结果"。
- 改投机 / 图模式 / RoPE 前先看 `Mean acceptance ratio`;`GPU prefill ACTIVE` 不等于已启用,要看 slack 正负。
- 结论写入台账(`docs/EXPERIMENTS.md`),不要只留在会话里。
