# B02 / K2.interface + K2.cpu 实现交付

日期：2026-11-20。状态：实现及适用 CPU 检查已完成，供主代理审阅/固定源码快照；没有执行、模型、GPU 或产品资格声明。唯一源码工作树为 `KernelSchema/.worktrees/schema-workflows-v1`，分支 `feat/schema-workflows-v1`。本代理未提交、推送或修改 Git 配置。

适用输入：完整读取 C（`planning/r5_contract_execution_review.md`）及 P/W/V 对应职责，消费 K1 材料固定提交 `35443d2f6cefe424e9adcd6d8d66dc17aa267d99`。共享 fixture 的 14 份原件均以 `git show <fixed-commit>:<path>` 实际核对摘要；不另写第二份 Task 判据。R0 仍是环境未绑定的材料，不能据结构检查称执行准入。

## 已交付代码

仅新增 `pyproject.toml`、`kernelschema/` 下本批 core 文件、`tests/workflows/`。未修改原 Template/extract 代码、原 Task/evaluator 材料、计划正文或其它仓库。

- 唯一 `kernelschema` CLI 与 Python `design/optimize/implement`，共同参数验证、请求/恢复定位和结果发布。原 `extract -- …` 启动同一原脚本的独立解释器，保留原参数及退出码。无已实现业务 backend 时在费用/创建 run 前明确 unsupported，不伪造结果或资格。
- `contracts.py`：单一整数域 RFC 8785 JCS、真正 UTF-16 键排序和控制字符编码；严格版本/字段/Ref/Source/预算/报告/结果类型；重复 JSON 键、NaN/Infinity、浮点 JSON、越界整数、BOM、surrogate 拒绝。小数由字符串表达，共同字节向量可独立复核。
- `artifacts.py`：通过目录描述符及 `O_NOFOLLOW` 读取相对 bundle 根的普通文件；严格稳定 publication prefix；Source 的 kernel.py 原字节/0644/size/hash/UTF-8 全快照和新目录原子安装；不以 overlay 补缺文件。
- 显式选根的语义闭包：Ref 字段、Source 实体、真实 Template 解析来源、Schema parent/change_report、原 extract delivery 的文件依赖；不扫普通文本、不复制无关历史。证据裸 hash 必须有显式原 request/Task/R/C/rules/method/evaluator 主体定位；跨主体错绑被拒绝。
- 不可变 Publication 持久分配 128 位 ID；同字节/模式幂等，冻结后新增/换闭包需新 ID。导入先核验/暂存全部选定依赖，再在本地 flock 下发布；已存在目标冲突或缺真依赖不污染目标。硬崩溃留下 durable import fence，所有 Bundle 读取须先显式 `recover_import` 回滚。文件与相关目录均 flush/fsync；未声称普通文件系统读者可以观察到多文件原子替换。
- `api.py`：三操作共用首份 result 前的 run 锚点、原请求/Q、backend/profile/implementation、owner 绑定；意图含原 operation/key/native locator/reservation，私有状态与终态持久接口；请求/操作/owner/实现损坏时不重开余额。result 规范历史和发布 ID 可在 projection 丢失后重建。
- `template_api.py` 以唯一 `kernelschema._template_tools` 私有名直接加载原 parser/normalizer/check/source trace；不复制 parser、不改 sys.path，也不加载两个裸 tools。非 root effective_document 包含 root Contract。
- `resources.py`：任意 cwd 的 checkout/editable 资源定位及真实规则/Template 源字节冻结；`materials.py`：明确角色 allowlist、目标唯一性、后端真实能力与硬 token 条件；私有 evaluator 文件不进入模型角色材料。原 extract CLI 所需的纯 CPU `jsonschema` 依赖已加入安装。
- `extract_bridge.py`：手写/curated/design/extract raw 共同消费，不要求统一 P6 门票；可保留原 receipt 字节及其全部声明真依赖，复核 receipt 覆盖所消费 raw/dispatch。历史初始化费用明确 unknown/分列，不伪装当前零费。

## 公共交接

详细、唯一源码侧说明为工作树 `kernelschema/INTERFACE.md`。公共入口包括：

- `contracts.{Ref,Check,loads,jcs,sha256,digest,validate}`。
- `artifacts.{Bundle,Publication,import_bundle,install_source,recover_import}`。
- `api.{Backend,RunContext,prepare_run,open_run,validate_result,design,optimize,implement}`。
- `materials.{for_role,validate_request,require_capabilities}`。
- `template_api.{tools,resolve_bundle,effective_document,normalize_directory,check_directory}`；`resources.{resource,rule_files,freeze_rules}`。
- `extract_bridge.{extract,consume_raw,implement_request}`。

K4 接口工厂为 `kernelschema.implement.reference.workflow_backend()`（backend=reference）；K5 为 `kernelschema.design/optimize.workflow_backend()`（backend=claude）。Backend 只需实现 `capabilities(profile)`、`identity(profile)`、`validate_resume(context)`、`execute(context,continuing=...)`。这不是通用插件/调度平台。K1.evaluator 使用唯一 `kind=evaluator/evaluation` 形状；K3 读取 profile、角色材料与 intent/native locator 接口；K4/K5 使用同一私有 state/persistence/result 接口。

C 未固定详细字段的 evaluator/profile/change_report/writing_review/feedback 形状在 `contracts.validate` 和 INTERFACE.md 明确；均为真实 Ref/subject 绑定，不以另一种 Schema 文本协议替代 C。`evaluation.environment` 是实际环境快照 Ref；optimize 行动使用 `action_reason/action_inputs/action_evidence`。

共同 fixture 位于 `tests/workflows/fixtures/v1/row_softmax/`：`build.py`、`k1_inputs.json`、`jcs_vectors.json`、README。生成器从固定 K1 原件按实际 hash 构造可移植、闭包完整的 synthetic bundle；支持三个请求形状，不提交假 hash 或不存在的请求。其 evaluator/profile 显式 synthetic/interface-only，业务入口应拒绝该未实现 backend。

## 实际验证

独立环境：`KernelSchema/.worktrees/.tooling/k2-cpu`，Python **3.13.12**；setuptools 84.2.1、PyYAML 6.0.3、jsonschema 4.26.0、pytest 9.2.0。依赖仅装在此 venv。没有冒称 Python 3.11 资格。

以该 venv 的 Python 执行：

| 命令/检查 | 结果 |
| --- | --- |
| `python -m pip install --no-build-isolation -e .` | 安装成功；唯一 CLI 可用 |
| `python -B -m pytest -q tests/workflows --maxfail=1` | **59 passed，23.40s** |
| 最后对字节读取/原 receipt 数值保留的定点修改后：`python -B -m pytest -q tests/workflows/test_artifacts.py tests/workflows/test_template_extract.py --maxfail=1` | **32 passed，11.67s** |
| 在 schema_template cwd：`python -B -m unittest tools.tests.test_cli tools.tests.test_design_resolution tools.tests.test_normalization` | **51 tests，OK** |
| checkout cwd：`python -B -m unittest discover -s schema_extract/tools/tests -p test_standalone.py` | **6 tests，OK** |
| 独立 cwd 子进程，禁止 AKA/torch/triton/CUDA import 后 import core、读取资源、加载 Template；editable `kernelschema --help` | 通过 |
| 原 extract 与透传的 `--help`/非法参数 stdout、stderr、退出码比较 | 分别 0/2 且完全一致；已确保原命令真实启动，不以共同依赖错误冒充通过 |
| 四种 Source 字节、非法路径/symlink/FIFO、错 hash/模式、额外文件、无关历史缺失、裸 hash 无主体、C1 证据错贴 C2 | 按预期往返或拒绝 |
| 真实 fork 子进程在第二个导入文件前 `os._exit(99)` | 留下持久 fence；已有/新 Bundle 均拒读；recover 后目标恢复原状并可重放 |
| 三操作首 result 前意图/owner 恢复、错误 operation/实现/原请求、相同 Q 显式新 run | 沿用原 run/操作或按预期拒绝；新 run ID/自有 owner 独立 |
| 固定 K1 14 原件及源码空白检查 | 全部匹配；无尾随空白 |

任意 cwd 生成及搬移闭包烟测的实际目录：
`KernelSchema/.worktrees/.tooling/k2-delivery-hu647qjy/`。
其中 bundle/ 与 relocated/ 均为真实可核 synthetic 输入；从 `/tmp` 调用 relocated 请求通过输入校验后以 `unsupported/not yet implemented`、退出 2 停止，未创建 run。没有新模型/GPU 调用或执行结果。

## 明确留给后续 owner 的工作

- K1：完整 evaluator/adapters、受信实际加载 C/evaluator、输入/副作用/数值/计时事实与执行资格。
- K3：真正 Claude 单调用、角色文件系统权限、实际模型/终态/用量资格。
- K4/K5：单 owner 私有账本/准入/活动时钟/结算、业务循环、seed ready/admitted 生命周期、Scope 的 Task 分区/when/variant 和执行语义检查、有限 fidelity/写作判定与业务恢复。K2 的 `prepare_run`/persistence 是工程接口，不是收费授权，也未复制账本。
- K6/K8/A 系列：有限 driver、真实产品/复杂题/AKA 接入及对应发行门；不由本批 CPU 测试代签。
- 已存在目标导入仅承诺通过 Bundle API 的本地 flock/fence 事务及恢复，不承诺任意并发外部写者或跨机私有 runtime 恢复；陌生 native 依赖协议需要其明确适配后才能宣称完整闭包。

本批源码已停止修改，可由主代理审阅并固定后交消费者；本报告不更新计划 gate 正文，也不替主代理签收或声明 KS-P1。
