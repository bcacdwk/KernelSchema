# Extract 的执行边界

当前入口为 [extract_yaml.py](extract_yaml.py)。Runner 使用当前 Template、P1–P6 提示、P2 有限选择、P4 写作安排及 P5 实际整区 `from`。命令见 [Supervisor 手册](README.md)，产物和批准协议见 [tools/README](tools/README.md)。

## 运行环境

需要 Linux、Python 3.11+、PyYAML、jsonschema、Git、`ps` 和已配置的 Claude Code 或 Codex CLI。Runner 与 CLI 都以当前用户身份直接访问真实路径，Extract 不再建立自己的进程/挂载隔离，也不要求 proc 挂载或 sudo 准备。

每个 scope 保留固定的真实空 cwd 与私有 CLI home；每次调用的 workspace、材料、stdin、原稿和规范化结果独立保存。调用前后核 cwd 为空，候选始终引用本次 attempt 路径，精确 resume 不依赖伪造 HOME/PWD 或目录镜像。

## 后端、权限与证据

| 范围 | 当前协议 |
| --- | --- |
| 默认阶段模型 | Claude Code `qwen3.8-max[1m]`，两角色 effort=`max`；参数与 effort 环境一致 |
| 可选后端 | Codex 明确登记模型和两角色 effort，使用自身原生协议 |
| 会话 | 每阶段/角色/操作新建，P5 按节点分开；同 scope 仅精确 UUID resume |
| worker | 本次 workspace 支持原生临时文件与原子 rename；最终文件类型、集合及字段差异核验 |
| reviewer | CLI 只读角色，独立有效全文审阅；补丁由 Runner 受控应用 |
| 进程 | 先登记启动门及身份；超时/取消确认全部所属进程终止，未确认时保持 fence |
| 成功结果 | 退出 0、明确原生成功、session、structured response、schema/nonce 和后续产物检查同时成立 |
| 用量 | requested/reported 分列，增量/累计口径分开，按 attempt 去重，缺项保持未知 |

worker 必须使用同目录临时文件才能完成 Claude 的原生 Edit/Write。暂存中已清理的临时名字不算产物；留下额外文件、目录、链接或保护内容变化的候选不能采纳。Claude 仅开启角色需要的 Read/Glob/Grep、worker 的指定文件 Edit/Write 和固定自检命令，禁止内部委派与定制扩展。Codex worker 使用原生 workspace-write 加本次 workspace，reviewer 使用 read-only；两者均关闭委派与自定义扩展。进程仍拥有当前用户身份；工具权限及候选差异校验不构成 Extract 对宿主操作系统的保护保证。

## Template 与六阶段

候选先核文件类型与许可集合，再保存原稿；复用 Template 做整体规范化，核字段权限与来源、落地严格重读，并交独立 reviewer。P1、P2、P4 使用阶段门禁；P3/P5 严格检查当前完成快照及解析必需来源；P6、发布和正式 standalone 使用完整目录 API。

P2 独立审定具体 root 和少量代表，P4 落实既定写作安排，P5 的内容共享由实际 raw `from` 决定。来源漂移按当前定义、Contract 和使用上下文核验；可证明等价时保存依据，否则复审受影响者。

裁决改稿也执行相同规范化与严格检查，然后由全新 reviewer 一次签收。当前维护保留固定身份、草稿、预算与未受影响批准；未完成阶段和最终 P6 仍须完成。发布核精确文件集合和当前批准，保留已审 raw，原子交付及冲突保护。

## 手工推演与阶段结论

单独的源码阅读、数学推演、候选评议或片段审查只声明实际覆盖对象。阶段完成需要本阶段材料、权限、检查和独立审阅；结构通过不证明语义正确，阶段批准也不证明源 kernel 的 GPU 正确性或性能。源码、任务或固定规则改变时另建运行。

## 验证范围

本次移除隔离的验证使用短离线/fake CLI，覆盖实际路径启动、角色参数、helper、原生结果、精确会话、恢复、审计与进程结清。只执行必要的极小真实 CLI 单步验证，没有重新运行真实 P1–P6 全流程或全套旧测试。当前结果与边界见[本次实施记录](../schema_discussion/extract_execution_20261117/isolation_removal/IMPLEMENTATION.md)。

此前完整重构的审阅与实测保留为历史证据，不代表当前执行方式的验证，见[原交付记录](../schema_discussion/extract_execution_20261117/FINAL.md)。
