# B12 — 最后交付核对（原计划 B11）

2026-11-20 UTC。此编号仅与刚结束的 B11 期限窄修复区分。**最终核对完成：KS `1c9e592c6e0ec021f91e17af5a8c39956ce39178` 与 AKA `804e26f79d094250a5852ad2b8a6f652f03f2928` 的两条有限工程路径、固定依赖、源码/包来源和证据边界一致，未发现待修的交付阻断。** 本报告不新增工作包、不安排真实调用，也不把工程交付写成真实任务全部通过。

本代理只写本文件及 `/tmp/ks-b12-final-audit-20261120-s7d0a0sg`。KS 源码、测试、公开说明及 AKA 均只读，没有安装活跃 AKA 树、重跑完整 suite、调用模型、提交 GPU job、派生代理或访问/枚举 EXP。所有命令限定于两个指定仓库、KS 固定工作树或明确 `/tmp`。原 main 的提取任务和 AGENTS 未改。已读适用 AGENTS、progress、delivery_audit、H048–H054、B10 两侧审阅、K5 真实 design、B11 期限和 B13 scope 配置报告；只读取明确的公开说明、必要入口源码、既有摘要/日志，没有递归展开私有 capture。

## 两个目标的工程结论

| 目标 | 本次核对结论 | 实际证据边界 |
| --- | --- | --- |
| KS 独立有限迭代 | 在固定 `1c9e592c6e0ec021f91e17af5a8c39956ce39178` 上，唯一 CLI/API、原 extract 透传、共享 design/optimize writer、reference implement 和实际有限 driver 均有源码与公开入口。Linux 源码/editable 是支持层。 | row-softmax 的实际 driver 和固定 FA4 F1 reference 工程成立；真实 design 尚未产出 Schema，optimize/GPU/fidelity/性能仍未资格。 |
| AKA 默认 off、开启后使用原 native 流程 | 最终 `804e26f79d094250a5852ad2b8a6f652f03f2928` 的 fixed/dynamic 工程、四包与原入口证据成立；KS pin/manifest、wheel 字节、公开说明、实际安装来源和不可变 probe scope 期限增量均已核对。 | 首发只支持 row-softmax full Task、single_file/kernel.py、Triton/never/no conversion、no-tools native profile；AKA F1 和 FA4 design 不在本次选择的路径内。 |

`kernelschema/cli.py` 直接调用同名 `kernelschema.api` 函数；extract 分支在 argparse 工作流解析前把原参数交 `extract_bridge.extract`，后者使用独立解释器调用原 `schema_extract/extract_yaml.py` 并保留退出语义。`design.py` 和 `optimize.py` 分别构造同一个 `WritingBackend`。API 只按显式 operation/backend 分派 reference、writing 或可选 AKA；独立 reference/writing 不要求导入 AKA。

row-softmax `driver.py` 真实调用 API，design 完成后进入 fixed；只有 completed fixed 才进入至多一次 optimize，实际执行合法 Source/default_start/verify 后续行动。同一 parent Ref 每步校验；新请求持久化与真正收费准入分开。普通 fail/unknown/耗尽不成为设计阻碍，尚无可信独立设计阻碍输入适配器。pending implement（包括同 R verify）清空 `current_qualified`，旧资格仍在 history。现行文档对此没有扩大承诺。

K8 采用 KS reference F1：有限完整 `python_tree`、固定 F1 覆盖点与 REST 未覆盖声明，不等于通用源码树/未知 FA4 变体支持。AKA 的有限接口并不阻碍所选 K8 工程路径交付。

## 阻断问题

**最终固定配对未发现待修的工程交付阻断。** probe scope 期限配置已按 supervisor 明确决定单独固定并通过其定点验证；本审计核对最终差异及工件，不另起修复、重复安装或大套测试。

真实门的缺失见下文，不能把它们从记录中删掉，也不应将已写好的入口、通道和有限编排描述为尚未实现。

## 已解决及本次确认

### 固定 KS、旧资产与公开说明

- KS HEAD 为 `1c9e592c6e0ec021f91e17af5a8c39956ce39178`，本次读取时工作树干净；该提交已由 supervisor 推送。前置最终工程修复为 `76ef9b1427e4d24174557054d049e7b134c9958d`。
- 以原 golden `2efc92479db497e0ecf681e276ecddc72cad0de6` 为基准，六个已有算子的 **21 份 YAML/Markdown**逐字节同时等于当前固定提交和原 main 实际文件。不是只比较文件数量，也没有覆盖其他正在提取的文件。
- `schema_extract`、`schema_template`、`EXTRACT_SCHEMA.md`、`kernel_wiki` 相对原 golden 的 Git diff 为空；原提取/Template 路径没有因 workflow 交付回退。
- 选择的 **12 份公开入口/接口文档、78 条本地文件链接**全部存在。公开命令使用明确 `$KS` 源码路径，说明 CLI exit 0 只表示结果已发布；默认无资格时为 blocked、零调用。安装说明明确 KS 资源来自 source/editable，并未声称完整 wheel。
- Git 固定树中没有 `kernelschema/runtime/qualification/`、`.runtime/` 或 `.claude/` 私有运行文件；`pyproject.toml` 排除 `kernelschema.runtime.qualification*` package discovery。该核对只读取 Git 路径和公开安装配置，没有展开被忽略的历史私有归档。
- 本次指出并由 AKA writer 修复了同版安装说明：四包仍为 `0.2.0`，普通 `pip install "$DIST"/*.whl` 可能跳过已有同版包。`e867333` 的 `schema-source-release.md` 已按实际安装使用 `--no-deps --force-reinstall`，明确先安装 KS editable 的 PyYAML/jsonschema 与 Framework/Legacy 的 `packaging>=23`，保留固定 KS 源码安装。无需改版本或再测 suite。

本次确定性产物：[KS 摘要](/tmp/ks-b12-final-audit-20261120-s7d0a0sg/ks-summary.json)、[六算子逐文件 SHA](/tmp/ks-b12-final-audit-20261120-s7d0a0sg/operator-preservation.json)、[公开链接清单](/tmp/ks-b12-final-audit-20261120-s7d0a0sg/public-links.json)。新增测试运行数、MODEL calls、GPU jobs 均为 **0**。

### 期限、恢复与证据的准确边界

真实 K5 design 在原 `76ef9b1` profile 的 120 秒内只有 init/thinking、没有 proposal，原进程已终止，reviewer 未启动。这一失败暴露的全产品硬 cap 已由 `1c9e592` 修复：默认 120000 ms 不变，业务 timeout 接受严格正 safe integer；实际 launch 仍取 profile、固定 request 和原预约紧限，UTC/monotonic 截止持久化。host 等待使用原截止加固定 5 秒终态收集，不从 continue 时重新领时。B11 没有更改原费用账户、默认 profile JSON 或旧失败 scope。

长业务 profile 是新身份，不自动继承旧 120 秒资格；本次不安排长调用或重试原 design。B11 配对阶段的 AKA probe 保持 exact 声明 profile，在原 call/deadline 层额外 cap 120 秒，没有拿短 profile capture 代签长 profile。随后新的真实 native probe 同样在原 120 秒超时，supervisor 明确决定再作最小配置修复，现已固定在 AKA `804e26f`：资格 scope ceiling 默认仍为 120 秒，但允许在新 scope 显式冻结；仍由 profile/request/owner 取紧，continue 只读原 scope 的不可变值。本次失败 scope 保持原 120 秒，不重跑；新配置的“可配置”不代签实际长调用。

原 76 公开源码快照的 183 文件、原 WritingBackend 身份匹配与原 handle 只读 reopen 已由 B11 保全，记录完整。本次复用该证据，不重新操作原失败账户或 private runtime。旧 bootstrap 在 core anchor 与 owner binding 之间中断/丢失时仍保守拒绝；已持久 operation 的恢复不能被解释成自动造新账户的许可。

本次指出 H051 把预览大小写成正式调用大小，supervisor 已校正为 **预览 354,691 / 正式 354,681 bytes**并在 progress 分列。K5 报告关于 120 秒全局 cap 的原观察保留为原 76 的历史事实；delivery_audit 已明确当前 1c9 修复，不追改旧失败记录。

### 复用的实测证据

| 已有证据 | 本次核对与可以支持的结论 |
| --- | --- |
| KS B10 完整 CPU | 原日志结尾为 **380 passed / 1504.60s**，属于修前 `8e846d7`；不写成最终源码完整重测。 |
| KS B10 修后 | 21 新回归分轮、56 兼容和最终账户组合丢失/原正向 4 项，按原报告保留；各组重叠不相加。 |
| KS 新目录 source smoke | 原摘要确认 128 文件独立 editable 子集、真实 prepare/run/continue/export、blocked/new_request、结果幂等、MODEL/GPU=0；它不证明完整发行 wheel 或 FA4 初始材料可匿名获取。 |
| KS 正式 FA4 工厂 smoke | 原摘要确认实际 reference/FA4 工厂准确因缺原栈资格 blocked；12 文件原 Source，continue 完整 result 与 usage Ref 一致、MODEL/GPU=0。 |
| KS B11 期限增量 | 原最终日志为 **9 passed / 76 deselected**、writing 兼容 **4 passed / 48 deselected**；首轮 34 通过和四项测试收据放错 bundle 的失败保留，修测试后定点复验。 |
| AKA 76 配对四包与 10 次原入口 | B10 原报告、wheel 摘要与安装路径记录可支持该次安装；原 workflow/optimize fixed+dynamic、KS→AKA 及各自 continue 全为 exit0/blocked，原 parent.operations={}、MODEL/GPU=0。最终 1c9 配对以追加部分为准。 |
| AKA 唯一完整 CPU 与 1c9 增量 | 已读原日志：76 配对安装完整 **155 passed / 1709.59s**；1c9 配对角色/probe/真实 fresh-child **26 passed / 65.20s**，原 receipt/unknown/V0/paid finish **5 passed / 35 deselected / 89.52s**。这些组不相加冒称最终全库重测。 |
| AKA B13 scope 配置 | 已读原日志：scope/probe **11 passed**、原 role/receipt/finish **10 passed / 51 deselected**、最后 owner/state 丢失修复及 CLI/receipt **3 passed**、最终安装 wheel 的真实 synthetic 子进程/来源 **3 passed**。末次 owner guard 在前两组后加入，其适用增量单列；不冒称全部在最后补丁后再跑。 |

实际日志/摘要为 `/tmp/ks-b10-full-cpu.log`、`/tmp/ks-b10-source-smoke-tsdqg7p4/summary.json`、`/tmp/ks-b10-fa4-smoke-u392ga40/summary.json`、`/tmp/ks-b11-runtime-deadline-20261120-YjaFav/recheck-runtime-tests.txt`、同目录 `writing-reserve-tests.txt`。本次读取上述确定文件，未借审计重复 suite 或另造演示。

## 必须诚实保留的缺项

1. **真实业务**：K5 原 planner 超时，Schema=null、四写作门 unknown；尚无真实完成 design/reviewer 发布、反馈 optimize、完整有限迭代成功。已实现业务入口不等于实际模型成功。更长 profile 的真实行为未测。
2. **真实 GPU/服务/栈**：GPU jobs=0；当前会话未取得可用 Agate 官方认证/所选 dev ACL 证据，原 SDK 缺签名头 401 不证明服务不存在。六作业、readonly/完整输出/cleanup、FA4 原 CuTe/SM103/driver/compiler/TVM-FFI、full-forward、独立 fidelity 和性能均须原实际证据。公共 timing 不能代替 native same-allocation ABBA。
3. **费用与原失败**：合计 **7 次**模型调用：K3 历史 4 次（含旧失败），已知 15,291 tokens、CLI reported USD `0.232672750000000010`，并非核实账单；旧 AKA coder 1 次、本轮 KS planner 1 次、新 B10 AKA coder 1 次均在各自原 120 秒范围内超时且 terminal usage 缺失。本次已核新 AKA 原 `observation.json`：实际输入 **355,182 bytes**、exit=-9、timed_out=true、process_alive=false，coder=1/reviewer=0/GPU=0，owner active=124139 ms、tokens/USD=null、qualification=null。总 tokens/费用未知，15,291 只是已知 token 下界；不改旧账、不把未知记零。
4. **支持与发行**：已测 Linux Python 3.13.12；Python 3.11 未测，KS 完整资源 wheel 未签。FA4 首次 prepare 的两固定 Git commit 匿名取得和项目许可证/完整第三方 notices 仍为发行缺项，保留 AKA 原 LICENSE/NOTICE不授予 KS/第三方新许可。它们不否认已完成的有限源码/editable 工程支持。
5. **明确范围**：AKA 仍只 row-softmax/single_file；KS writing/driver 仍只 row-softmax，FA4 F1 仅 reference implement。本次不扩 AKA F1、FA4 design、任意源码树、所有 case/backend 组合或恶意 native CLI 沙箱。

K1.real、K4.real、完整 K5.real、K6.demo/release、K8.real、A2.real、A5.real、A6.real、A7.fixed/pair 不得因本次源码交付核对改成真实 completed。最终结论应同时交付可用工程路径和上述具体证据边界。

## AKA 永久锚点与最后配置差异

`e867333dfcbbce553465635bcf57a9bde842b06f` 已由 supervisor 精确提交。本次直接读取该 Git commit，没有导入活跃 editable AKA 或安装任何包：

- gitlink 与实际 detached/干净子模块均为 KS `1c9e592c6e0ec021f91e17af5a8c39956ce39178`。`release.json` SHA-256 为 **`f85aa4e254407d319318520aab49b7a448b4d77dd7a7551baffee490d78fb14e`**；明确选择的 **297** 个源码/资源文件逐项 hash/mode 一致，包含 core/runtime、规则、Template/extract、row-softmax 资源和明确 fixture，不包含私有资格归档。
- 四个已测 wheel 内 **492/72/78/28** 个 AKA 文件全部以 Git blob 字节身份核对 `e867333`，无 KS 或私有 runtime 入包。每包 dist-info 的 LICENSE/NOTICE 与原 upstream `bf5557ac04f6ff9988901d511cf25e1cbe61a1bc` 一致。Schema wheel SHA-256 为 **`69962cda200cfd0240d686128642223f084eb2b21235d440eca5b76ce09e4d4f`**，其余三包 SHA 保留在本次 JSON 摘要和 B10 报告。
- 原失败安装归档 `/tmp/schema-b10-aka-native-failed-release-20261120` 的 **680** 个选定文件逐项 hash/size 和不可写属性匹配；`INDEX.json` SHA-256 **`01260d28efd3ed82191d48f796289e66c3ff78e25c86a45ade6bb03db21769a2`**。其中 `aka_input_head=04fab50` 是修改前输入身份；实际源码/wheel 与提交后的 e867333 相等，由本次独立比较补足，不能只凭旧 HEAD 字段误认运行代码。
- 已读 `/tmp/schema-b10-aka-installed-imports-final.json`：真实四包当时都来自 Python3.13 site-packages，KS 模块与资源根均指向该固定子模块，manifest 同为 f85。随后 writer 将 Schema 恢复为 editable 开发环境的动作单独记载；本次没有把此刻活跃导入冒称原实际 wheel 安装。
- 已读原完整/定点日志、10 次入口 smoke 摘要及 off 比较：上游/默认 off 归一化 trace 相等；10 次 CLI/continue 属于 76 配对，最终 1c9 增量另外覆盖角色、probe、实际 fresh-child 和付费 finishing 恢复。没有把测试来源版本混成一轮。
- 源码中原 CLI 仅在 `options.schema` 真时导入 optional launch；默认/off 忽略 schema-config。固定 factory 空参数进入 `FixedController`，显式 dynamic 才调用共享 KS writing；KS→AKA 路径复用同一个 controller，不再嵌套 implement loop。新 Q/R、同 parent、D3、pending 资格和原 finishing 语义与两侧报告一致。
- 固定提交的 **8 份**公开 Schema 说明、**27 条**本地链接全部可解析；同版升级修正已实核。公开说明保留有限支持、no-tools 的实际权限、原私有账户恢复边界和所有真实门。

确定性核对结果：[AKA 固定锚点摘要](/tmp/ks-b12-final-audit-20261120-s7d0a0sg/aka-anchor-summary.json)、[AKA 公开链接](/tmp/ks-b12-final-audit-20261120-s7d0a0sg/aka-public-links.json)。这些只是字节/来源核对，本次仍 **0 测试 suite、0 模型、0 GPU**。

### 最终 `804e26f` 差异签收

上述锚点记录的“probe 固定 120 秒”产品配置限制已由 [B13](b13_probe_scope_limits.md) 修复。最终提交为 **`804e26f79d094250a5852ad2b8a6f652f03f2928`**，本次核验工作树干净。相对 e867333 只有明确的 **8 个文件**：两份公开说明、native_probe/agate/session/stdin_exec 四个源码文件和两个既有测试文件。每项 Git 字节、size、mode 与 `/tmp/schema-b13-source-evidence-final.json` 一致；KS gitlink、release manifest 和其余三个 AKA 包源码没有变化。

新 `--scope-timeout-ms` 默认 `min(120000, declared_profile.timeout_ms)`；显式值必须为不超过声明 profile 的严格正 safe integer。新 scope 的 backend identity 在创建 run 前校验并冻结值；transaction binding、每 operation ceiling、locator 和原 exec `locator_sha256` 共同绑定原值。原 profile/hash、request、同 parent、原预算/绝对截止和 USD threshold 不改变。continue 拒绝 override、不回填默认；公共 `capture(context)` 即使漏传 continuing，遇已有 state 也只打开原账，原 state/budget/owner/intent/terminal 任一残留缺配对时保守拒绝建新账户。资格消费核原 scope、exact profile、locator/exec 关联，缺失或篡改不能签收。业务 RoleSession 默认 `None` 的 ceiling 行为保持。

新增 CPU 组使用原 NativeSession/OwnerAccount 与明确 synthetic transport，验证显式 300 秒 ceiling 可到达原 process 接口、原 native 180 秒/request 90 秒进一步取紧，缺绑定和篡改拒绝、原 receipt 不重调、实际新进程经过原 stdin_exec。不是实际等待 300 秒或真实模型资格。原失败 scope 未继续、重试、换 owner 或修改原限额。

本次逐项核对 **9 个**已记录日志/安装来源/包工件的 hash 和 size，实际日志结果与上表一致。最终 Schema wheel：

`/tmp/schema-b13-aka-dist-final/atrex_aka_schema-0.2.0-py3-none-any.whl`

SHA-256：**`5e4948e973242ef2b4e72388d63c2d71c1ee8c00735a11d2959bf47cfebafa0f`**。

其 **28 个** AKA 文件逐字节等于 804e26f；原 LICENSE/NOTICE 相同，无 KS/private runtime 入包。最终 sdist 和 index 的摘要与 B13 索引一致。原 Schema wheel 69962cda 和失败归档 INDEX 01260d28 均仍保留，新轮没有覆盖原证据。

`/tmp/schema-b13-installed-provenance-final.json` 记录了从 `/tmp` 的 `python -I -B` 新进程：四包当时全部从 site-packages 导入；所列模块 hash 与最终 Git 一致，KS module/resource 仍为固定 1c9，manifest 仍为 f85。最终 wheel 上的 3 个新进程测试通过。之后恢复 editable 的路径有独立记录；本审计未把恢复后的 editable 冒称 wheel，也未运行活跃源码。最后两份改动文档的 11 条本地链接有效，300 秒示例明确先有足够声明 profile/原余额，continue 不带 override。

本次最终摘要：[aka-final-summary.json](/tmp/ks-b12-final-audit-20261120-s7d0a0sg/aka-final-summary.json)。结论为两个有限工程目标可交付；真实门仍按前述缺项，累计真实调用保持 **7**、总 tokens/费用 **unknown**、GPU **0**。本审计自身没有新增测试 suite、模型或 GPU；不再提出超出本次范围的新工作。
