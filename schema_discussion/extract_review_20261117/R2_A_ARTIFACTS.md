# R2 A：阶段产物、来源与 Template 接口

基线 `be4fc48`，保留 R1 已有改动。本轮阅读当前六阶段说明、执行协议、R1 A、Template API，以及产物、输入冻结、来源、发布和 standalone 路径；只在授权文件内修复确认问题。未执行真实模型、queue、GPU、提取批次或正式发布。

## 确认问题与修复

1. **P1/P2 的适用文本检查不完整，warning 又被升级为错误。** `artifacts.py:_early_check` 原先只检查结构和算法区文本预算。合成例的 `Name` 51 字、`Describe` 151 字或 `Dispatch.when` 501 字均能通过早期检查；共享元数据随后会在完整 Design 检查失败，P3 又无权修正。另一方面，算法描述结尾句号的 Template warning 会直接阻断 P1/P2。
   - 继续复用 `rules.validate_algorithm`，其余当前内容交现有 `text_policy.check_values`；不复制字符常量，不要求未来 Design，不扩大硬件检查。
   - 按 Template 的 `Diagnostic.severity` 区分错误与提示，早期检查和 P2 只读 helper 都保留 `notices`。
   - 核对了其他早期允许字段：算法已有结构、关联和值预算；Source 仍绑定冻结来源；Execution 仍复用 `named(known_text, key=hardware_identity)` 核最小登记，其他硬件检查与预算按 `skip-hardware` 保持豁免。
2. **固定指导已被正文引用，却不在规则快照中。** `rule_files` 漏掉 `schema_extract/EXECUTOR_ALIGNMENT.md` 和 `schema_template/tools/README.md`，现补入冻结清单。B 已同步材料导航和相关说明；没有递归引入 wiki、案例、archive 或旧报告。
3. **standalone 会把旧校验器运行结果归到新 Template 代码身份。** 最小复现：在临时复制的 Template 中生成一次视图，随后更新 `schema.py`，让新 validator 明确拒绝；同一进程第二次 `create_view` 仍成功，回执却记录新代码摘要。
   - 使用既有 Template manifest 的 digest 区分解析包缓存；冻结 run 继续按其不可变根加载。
   - 生成改用同一解析包的 `schema.effective_yaml`，随后仍严格调用 `validate_standalone_file`，生成、校验和回执绑定同一代码身份；提交前再核一次该 manifest。
   - 保留实际内容 `source_files` 与完整目录 `validated_directory_files` 的区别，仍检查跨区来源闭包、原件漂移、冲突及复用。没有增加锁、监视或兼容路径。
4. **工具测试排除条件误用了绝对路径。** Template 代码收集在仓库的某个祖先目录恰名 `tests` 时会排除所有代码，导致冻结包缺失或 standalone 的代码 manifest 为空。两处均改为检查仓库相对路径；上述同进程更新回归特意把临时规则放在 `tests/current-rules`，同时核对冻结清单包含真实 validator。

## 必要离线验证

全部测试使用 `PATH=/usr/bin:/bin /opt/venv/python3.13.12/bin/python3 -B`；真实 claude/codex 不在测试 PATH。材料、规则更新反例和视图均位于临时目录，helper 流程使用确定性回执。没有修改真实 Template 文件。

- 修复前，两项新增早期检查测试产生 8 个失败子场景；新增 standalone 更新规则测试失败于“应拒绝却成功”。
- 修复后，早期两项与 `test_source_gaps` 共 7 项通过，覆盖允许的来源缺项、明确比较器事实和 nullable IO；未来 Design 仍可未完成。
- `test_standalone` 6 项通过，覆盖实际来源、完整目录、精确 Decimal、节点自身内容、来源漂移、冲突/复用和当前 CLI。相对路径过滤收尾后，更新规则回归再次通过。
- P1 原稿/规范化、Execution 写入边界、共享元数据派生及跨区解析闭包四项现有回归通过；同组再次运行更新规则回归，共 5 项通过。
- P1 只读 preview 与 P2/P4 helper 两项通过；同组对最终版本的更新规则回归通过，共 3 项。
- 变动 Python 文件 `ast.parse`、规则清单的指导/代码包含与工具测试排除断言、`git diff --check` 均通过。所有测试进程已退出。

本轮未改发布控制或批准逻辑。代码复核保留了各阶段完成范围、原稿与规范化记录、字段权限、checkpoint、共享 Contract、实际采用来源和额外解析闭包的现有区分；最终目录和发布继续使用完整检查，不据本轮离线测试声明真实模型质量或真实全流程已验证。

## 变动文件与未决项

- `schema_extract/tools/artifacts.py`
- `schema_extract/tools/reading.py`
- `schema_extract/tools/snapshots.py`
- `schema_extract/tools/standalone.py`
- `schema_extract/tools/tests/test_artifact_transactions.py`
- `schema_extract/tools/tests/test_standalone.py`
- 本报告

没有留下本轮已确认而未修的产品问题。产品文案及导航由 B 同步；未修改 `publication.py`、`inputs.py`、`comparators.py`、Template、Schema 字段规则或 Kernel，未提交 Git commit。已停止产品写入。
