# 六算子全量提取：Supervisor 交接

任务仍在进行，完成条件为六项全部 P1–P6 通过并正式发布，再核对产物、审计和 issue。不是小测试，不因本次提醒结束而停止提取。

- 当前 repo：`/home/shaohanyong.shy/KernelSchema`，主线程 `01a1c900-a487-76a1-92ee-990cbff08276`
- 六项身份、输出、最近观察见 [RUNS.json](RUNS.json)，过程见 [PROGRESS.md](PROGRESS.md)，问题见 [ISSUES.md](ISSUES.md)
- 本批 `.schema_extract/v2`，固定源提交 `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2`、冻结执行器 `b8a1d5e...`；worker/reviewer 使用 Claude Code `qwen3.8-max[1m]` / max
- 旧运行整体在 `.schema_extract/archive/20261118-before-six-rerun/`，不要恢复；历史稿与映射见 [ARCHIVE_INPUTS.md](ARCHIVE_INPUTS.md)
- 当前阶段、在途和终止状态以公开 `inspect/status` 为准；停止后先确认结清，再按已有授权与原计数恢复，不篡改状态、回执、稿件或冻结快照
- 本批 EOF 回执越界已确认为 Claude Read 显示虚拟末行，保持真实范围校验；普通已停 P1/P2 使用公开 whole-run resume 续精确会话，先验退出/终止/会话/余额，命令日志见 [RECOVERIES.jsonl](RECOVERIES.jsonl)
- MLA 有 `tuple-abi-input-typing` 两条 Input 的限域维护；任务见 [MLA_TUPLE_MAINTENANCE.json](MLA_TUPLE_MAINTENANCE.json)。维护verify的恢复也用 whole-run resume；外层游标P2，不能用 --scope P1。核对记录见 [MLA_MAINTENANCE_RESUME.md](MLA_MAINTENANCE_RESUME.md)
- 不修改已固定的 Schema 字段规则。仅收集的说明建议 `non-tensor-parameter-registration` 非阻塞；tuple规则扩展主张已分流，当前实例修正仍须Qwen独立复核

当前源码已修：无维护请求时的空转取锁、scoped resume的拒因反馈、可修回执错误分类、长文本规范化警告上下文，以及monitor登记取锁。前四项不热更新本批冻结工具；monitor的下一次新CLI请求使用当前源码。各报告在本目录，均做了定向验证，未提交或推送。

收到提醒严格按对应 ack/scan/status 协议办理，确认 committed/reused 与 next_due_at；ack后若scan未完成，只续同一scan。不要另建线程或闹钟，也不以日志沉默直接终止调用。后台观察与Supervisor恢复各自独立。

遇到新的实质工具问题用新xhigh代理定位和最小修复；同时至多两个代理，普通核查可用high。没有新问题不扩大审计或重测。按阶段进展读取candidates，reviewer确认项按根因查重登记；六项全部交付后做收口，并停止本批提醒。

15:30追加：三个调用者输出缓冲的P1限域维护已登记：Sparse `5677bbf8-286e-4169-b91e-59aa0f8e6445`、MoE `4d304fe6-bfdf-49ab-9b1f-5c861b5c6f8c`、FMHA `1170387d-57bb-483e-9ffa-1e960c39ab2c`，任务为各 *_OUTPUT_BUFFER_MAINTENANCE.json。前两项已review，FMHA当时draining。维护结束后MoE仍可能须处理原P4回执拒因。GEMM/GDN已各进入单个P5节点（wide/segmented）；P5出现局部失败时先按实际健康兄弟与冻结恢复协议判断，不盲目整项恢复。维护命令busy可能是登记后delivery争锁，须用inspect和同载荷幂等重放核实。

19:57追加：MoE split_k因入口证据缺项失败，但mega仍健康在途。独立xhigh确认本冻结版本的scoped恢复无可靠拒因交接，因此未执行局部恢复；请待所有active/unsettled/unconfirmed清空且Runner退出后，核对新状态，再用公开whole-run恢复失败节点。不能为赶进度中断mega。详见MOE_PARTIAL_RECOVERY.md。GEMM已正式发布并由high只读核对，MLA亦已交付，仍需四项完成后整体收口。

**04:26最新关键变更**：Sparse旧run因P6 inventory工具缺陷无公开恢复路径，已仅该算子新建successor `a362b85a-dce6-4ad4-b38e-3a186dd87c1c`（当前修复executor b6c8be…，P1 Qwen/max，全部批准重走，旧完整稿只读参考）；勿再恢复旧Sparse。RUNS.json的runs已替换，旧身份在superseded_runs；详细SPARSE_SUCCESSOR_LINEAGE.json。同一monitor已stop确认旧进程退出后start，generation3，仍check43、原next_due不变；新提醒check-3-44-c940af3ecb2ee6e064824770。新run先发布sparse_successor_delivery，再用public export交标准schema_wiki原目标，不能直接搬文件。root寄存器保证须按REGISTER_CLAIM_TRIAGE在新P3正常纠正。当前仍4/6已交付，FMHA P6意见定点返回P3处理中；没有重建其他五项。

后续inspect请显式传RUNS.json的runs内六UUID，或过滤其集合；Store中现在实际有七个run，旧Sparse保留用于审计，不在活跃任务列表内，不得因其仍invalid_receipt而自动恢复。所有批次时间/用量汇总须同时计入superseded_runs的真实消耗。

**最新讨论约束（2026-11-19T04:48:48.415407+00:00）**：用户质疑从P1重跑的必要性和效率。Supervisor重新判断：旧4稿/目录基本完成，不能把缺少恢复接口等同内容必须重做。已公开pause successor a362b85a，request fe4c626f-76d7-4e01-b774-212b44666d77/control_revision1，在途P1可自然结算，不启动下一调用。不要自动resume新Sparse、恢复旧Sparse或再启动重建；等待保进度恢复方案进一步敲定。其他算子继续，FMHA未暂停。没有修改goal状态或任何审批。
