# Extract 真实全流程验收

本轮已停止，统一汇总见 [ATTEMPT_REPORT.md](ATTEMPT_REPORT.md)。两个算子尚未完成P6或正式发布。


基线：`fa853fa`；开始：2026-11-17 UTC。

并行提取四题固定来源中的 FA4 Prefill 与 QSA Prefill。阶段 worker/reviewer 使用 Claude Code `qwen3.8-max[1m]`，effort 为 `max`。沿正式 P1–P6、独立审阅、Template 检查和发布流程推进，不运行 GPU。

来源绑定固定到 `fa853fafd32a78ac496fbf71e615e34200eeeae3` 的四题源码树，各任务明确独立入口、IO、输入、reference、验收和范围材料。FA4 的历史网关结果与 QSA 的验收器分别绑定。

初始预算：每算子 21600 秒，每阶段 3600 秒，每调用 1800 秒，每 scope 32 次调用；worker/reviewer 维持默认 `max`。

状态目录：仓库下 `.schema_extract/live_acceptance_20261117`。来源描述保存在本目录 `sources/`；每次运行使用独立的新交付目录。运行身份及结果更新到 `RUNS.md`。

发现工具问题时保留原始运行证据，修正当前工具并做定向验证；涉及冻结规则或执行器变更时另建 run。批准与交付由正式 Runner 完成。

验收要求：两个算子均完成 P6，状态为 `ready` 且 `published: true`；交付目录、独立结构检查和审计证据一致。

## 问题

| ID | 发现 | 处理状态 |
| --- | --- | --- |
| LIVE-01 | 已删除的 `schema_extract/EXECUTOR_ALIGNMENT.md` 仍被手册和规则快照清单引用 | 已修复；冻结及材料导航验证通过，见 [LIVE01](LIVE01.md) |
| LIVE-02 | 两个 P1 首次 worker 均约 15 分钟被时间预算终止；配置单次上限为 1800 秒 | 阶段内审阅/裁决/签收预留导致有效首稿预算 900 秒；非计算漏洞。已通过正式恢复各追加 7200 秒，详见 [LIVE02](LIVE02.md) |
| LIVE-03 | 可选示例导航落空；QSA 主动查阅检查器规则 | 必需规则齐全，未发现规则冲突；只记录观察，见 [LIVE03](LIVE03.md) |
| LIVE-04 | 恢复后两次 P1 调用 1800 秒仍未写稿；FA4 原生 CLI 插入输出上限续写提示 | 核实默认32K、adaptive/max；FA4命中上限而QSA未命中，见 [LIVE04](LIVE04.md) |
| LIVE-05 | 需要在真实运行中显式指定 Claude 原生响应预算 | 新增 `--max-output-tokens` 并冻结/审计；40 项回归和一次真实 64K 短请求通过，见 [LIVE05](LIVE05.md) |
| LIVE-06 | 无效配置在报错前占用 output 登记，修正参数后无法复用未创建的目录 | 配置校验移到新登记前；同路径无效→有效及冻结冲突验证通过，见 [LIVE06](LIVE06.md) |
| LIVE-07 | worker 规范化预览通过但因待格式调整返回失败码 | P1–P5 worker 改按预览结果返回；审阅/P6不变，4 项测试及真实草稿副本验证通过，见 [LIVE07](LIVE07.md) |
| LIVE-08 | 验收器 Python 源码含 relative_to 被强制当 JSON；回执另有输出键误填 | 按真实资料格式处理，保持 JSON 严核；补清 comparisons 键名约定，8 项测试通过，见 [LIVE08](LIVE08.md) |
| LIVE-09 | 回执拒绝后丢失已完成原生会话UUID，启动前错误误成终止未确认 | 会话登记独立于提案批准；复用未启动结清协议，8项定向测试通过；旧现场保留，见 [LIVE09](LIVE09.md) |
| LIVE-10 | QSA旧冻结版本的P2拟批准回执含超长证据，普通恢复会触发旧UUID问题 | 真实冻结代码临时副本验证正式pause→maintain review→resume可首派新独立reviewer；不新增代码，见 [LIVE10](LIVE10.md) |
| LIVE-11 | 旧QSA维护invalid_receipt进入blocked后无正式重试入口 | 临时副本确认所有旧入口不能恢复，原现场保留，见 [LIVE11](LIVE11.md) |
| LIVE-12 | FA4 Runner消失，CLI已退出但无原生完成结果 | 冻结正式resume可结清为中断并重新独审，反例fence保留；实际后台恢复成功，见 [LIVE12](LIVE12.md) |
| LIVE-13 | 当前维护回执无效无法正规修正，证据错误定位不够明确 | 既有resume恢复受限normal任务；精确报告索引/文件/行数，8项测试通过，见 [LIVE13](LIVE13.md) |
| LIVE-14 | 内部固定12行证据范围反复拒绝真实有效片段，必要性需复核 | 独立数据评估684条证据，建议取消任意选择跨度上限，保留真实性与独审，见 [LIVE14](LIVE14.md) |
| LIVE-15 | 删除不提供真实性保证的任意选择跨度门槛 | 两处条件删除，10项定向测试通过；21条纯长度拒绝消失、2条真实越界仍拒，见 [LIVE15](LIVE15.md) |
