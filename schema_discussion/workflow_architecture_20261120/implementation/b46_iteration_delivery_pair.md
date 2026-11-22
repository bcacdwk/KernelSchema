# B46：有限迭代的源码配对与独立边界核对

2026-11-21 UTC。**B43/B44 新入口与环境兼容已做有限独立核对；具名子模块/release 已配到 KS `13efd1554d112d56e4d767ba43b66257b2d75669`。另外按根授权修复一个实际未结算结果透传缺口，六项定点 CPU 通过。** AKA 产品变更已按精确 manifest 由根固定为 `88d773b3acdd60ed13334dc46404e2d0caefe49f`，本 agent 未 stage/commit/push；最终push回执待根确认。B45 真实有限运行尚待根移交终态，本报告不提前签新 candidate 或性能结果。

本 agent 外部模型/GPU/安装/服务启动均为 0；未派生子代理。只修改指定 AKA 和 KS main 的本 discussion；KS 独立工作树产品文件没有改动。未读取、枚举、运行或修改禁止实验仓及 archive 变体，未从 home 递归搜索，未查看或操作原真实账户/进程/运行文件。

## 源码配对与公开说明

- 起点为 KS `13efd1554d112d56e4d767ba43b66257b2d75669` / AKA `30bd9554dfb11d874a9b0dbaaab46748845ae79f`，同分支 `feat/schema-workflows-v1`，两个产品工作树初始 clean。KS 已由根推送；AKA最终配对提交随后由根确认为 `88d773b3acdd60ed13334dc46404e2d0caefe49f`。
- 按授权从 KS 指定 worktree fetch 同一个 Git 对象到既有 `optional/KernelSchema`，随后 detached checkout 到 `13efd155...`。只处理这个具名子模块，未初始化其它 submodule。子模块仍 clean，原 `origin=https://github.com/bcacdwk/KernelSchema.git` 未改；AKA origin 仍为 alibaba，schema-fork 仍为 bcacdwk。
- 使用原 `python tools/freeze_schema_release.py --commit 13efd1554d112d56e4d767ba43b66257b2d75669`，生成 **301 个选定文件**。`release.json` SHA-256 为 `9d2c4f4789b75b4e9e2f7c791cbf0a7965d3768137bbcc54739a5e9c8e1e8da4`，包含新的 `kernelschema/iterations.py`。原 `aka_base_commit` 仍是历史 integration base，不冒称当前 AKA 包身份。
- 原 `verify_release()` 在现有 B18 Python 和 source PYTHONPATH 下通过：KS 实际来自具名子模块，版本 `0.2.0` 的 AKA Schema metadata 可读，manifest/hash/mode 相符。此为来源核对，不给 standard 新增 legacy 门，也不是新 wheel 安装测试。
- AKA standard-workflow、optional-module 的当前固定 KS 链接更新到 `13efd155...`；补充当前 typed native 测量在 environment identity=null 时的准确边界，以及显式 owner 的 `usage.unsettled` 含义。两篇 KS 入口/示例文档已准确说明版本粒度，无需为本次配对改 KS。
- B40 公开 root 仍为 **9,106 B / SHA-256 `58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c`**。其 README 的旧 `907c99...` 是生成该原稿的真实来源，特意保留，不误替换成当前运行版本。

## 必要修复：原阶段未结算事实必须到达循环

B45 的零费用 fixture 在 `StandardBudget.model` 原回调于收据返回前抛出 `RuntimeError`，原账户正确记 `terminated=false` / `uncertain`，但 `standard.implement` 只返回 `failed` 且没有 usage，KS `iterate` 因而仍调用了一次 optimize。根确认这是收费前必须修的小缺口并授权。

仅增加 **15 行产品代码**：

- `StandardBudget.unsettled()` 使用原 `account._read()`，按同一 run_id 和 workspace+Schema identity 所得原 group_id，筛出本 stage 未 terminated/released 或 uncertain 的 operation IDs。没有调用会 tick/写账的 `operations()`、`fences()` 或 `usage()`；未创建账户、预约、结算、重放或新状态机。
- `standard.implement` 仅在真实 selection 配置显式启用 owner 时读出该字段，写入 `result.usage={scope: native_stage, unsettled: [...]}`。保留原 status、exit_code、RuntimeError 和 stop_reason，KS 已有判断据此在下个 writer 前停止。默认无 owner 路径保持，历史 tokens unknown 本身不会被当作未结算。

回归 fixture 按真实 selection 的 owner/workspace/schema identity 构造原 StandardBudget。B45 初始探针未将其独立 synthetic owner 传入 API，且用了单独 identity；这只能证明漏传场景，不能要求默认无 owner 的接口去发现未声明账户。正式回归绑定实际配置，并逐字确认结果投影前后的账户及已有收据未改变。

## 有限独立检查

| 范围 | 本次结论与证据 |
| --- | --- |
| 新入口与默认值 | AKA 薄适配直接调用唯一 KS `api.iterate`；传原 WorkflowLaunch、provided settings/runtime、ProblemSource、workspace_base、patches。standard 每版只换精确 workspace/name、清旧 restart arguments。默认两版本至多一次 optimize；版内 max_iters 仍是原 coder 轮数。 |
| 包含与安装 | 静态核对 KS `kernelschema*` finder、console entry；AKA 原 builder 的 schema目录 ownership/py-json-md资源包含新 iterations、standard_budget、release。现行文档安装匹配 KS source/editable 和四个 AKA 包。B45 承担实际 help/包检查，未重复安装或构建。 |
| 真实失败与停止 | 原 native JSON/声明反馈按字节保留；settled failed 可交 optimize，blocked/interrupted/unsettled/缺反馈停止；write-only未completed/无产物停止，completed keep停止；已有任何输出目录拒绝自动重放。新 budget结果修复补上实际漏传，未扩大其它 gate。 |
| 环境未知 | 窄读 B44 四处产品 diff 与原 source/measurement/terminal/SDK ABBA校验。当前 typed单次测量与同allocation完整ABBA可使用实际结果，identity仍null/comparability仍unknown；跨记录reuse/minima保持原拒绝，显式unsupported和SSH未知能力仍拒绝。没有改CandidateGate或codec。 |
| CPU新反例 | `tests/test_schema_iterations.py -k 'original_native_budget or stage_usage_excludes'`：**4 passed, 7 deselected in 1.46s**。缺model收据、pending GPU都writer0；settled RuntimeError仍writer1；同父另一group不污染，读前后账户/收据bytes相同。 |
| 原默认兼容 | `test_native_failure_returns_recorded_feedback` 与 `test_completed_api_call_reuses_exact_result`：**2 passed in 0.76s**。原无owner失败/缓存返回保持。 |
| 复用而非重跑 | B43 KS27、AKA21+最后2项，B44 28项CPU按各自报告范围复用；本次未跑旧strict大套、Schema效果或新kernel验收。 |
| 静态检查 | 三个改动Python AST、精确diff空白、root bytes/SHA及原release verifier均过。文档链接与历史保留检查另记录于本地audit-verification。 |

CPU 使用既有 `/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python`，`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 与明确的两仓 source PYTHONPATH；没有安装或热改 B45 环境。上述 fixture 明确 synthetic，不能替代真实候选/full/gate/ABBA。

## 产品 manifest 与本地审计分开

产品清单为 [source-manifest.json](/tmp/schema-b46-iteration-delivery-pair-20261121/source-manifest.json)，SHA-256 `1419a411228423cf2e2eb97f670280c2b2d10e9bbb4d4d684a9a27c7e0d0a8b2`；[aka-source.patch](/tmp/schema-b46-iteration-delivery-pair-20261121/aka-source.patch)、[frozen-aka](/tmp/schema-b46-iteration-delivery-pair-20261121/frozen-aka)、[verification.json](/tmp/schema-b46-iteration-delivery-pair-20261121/verification.json) 保存精确副本及检查。

仅 AKA 六个普通文件加一个 gitlink：`docs/schema-optional-module.md`、`docs/schema-standard-workflow.md`、`src/aka/workflow/schema/release.json`、`src/aka/workflow/schema/standard.py`、`src/aka/workflow/schema/standard_budget.py`、`tests/test_schema_iterations.py`、`optional/KernelSchema`。交接时父index仍旧907c99，根需显式add gitlink；测试已在B43被显式加入Git，本次是已跟踪文件。KS产品修改0。没有混入本地报告或真实运行原件。

本地审计只改本指定 discussion 的 `implementation/README.md`、`progress.json`、`next_steps.md`、`delivery_audit.md`、本报告以及 `01_history.md`。修改前副本在 [before-audit](/tmp/schema-b46-iteration-delivery-pair-20261121/before-audit)；后续审计清单独立于产品manifest。B01–B13原文、progress历史units/gates/old rollup和H001–H090前缀保留。

## 真实结果边界与待根移交事项

已据 B41 原报告收口：环境身份缺失导致根授权停止第一调用；`invalid_handoff`、candidate=null、changed_paths=[]，只有原reference V0三shapePASS，没有候选full、CandidateGate或ABBA。**1 host model invocation / 1 accepted V0 job / 661725ms**；partial tokens **239774**，完整tokens/USD unknown。父累计 **8 models / 7 jobs / 3977366ms / 28 operations**，active=null、unterminated=[]、uncertain=[]；B41 stage和父差1ms按原整数结算保留。

B40真实write-only仍completed/revise，planner1/reviewer0/GPU0，40303 tokens、native USD约0.58199875；B33旧API failed、B35/B36无候选原失败与费用未知不改。当前修复后的B41只读投影仍是reference，不能改称新candidate。

B45拟从B40原Schema通过公开 `api.iterate` 进行最多两版本标准native和中间一次optimize；revise才进入第二版、keep停止。根拟定新parent request最多coder/models2、jobs8、active4800000ms，各版原group1model/4jobs、max_iters1/handoff0、agent1800s/episode2100s；writer独立profile600s/API610000ms/USD2、reviewer0。根已在固定KS13efd/AKA88d773后授权B45唯一收费slot，先做四包/help/preflight，通过即执行上述有限iterate。B45用真实selection配置独立确认writer=0、iteration=blocked、原unknown operation保持；初始未配置probe只作发现历史。**授权不代表实际调用、注册或成功。** B45真实终态与完整结算由根提供后再追加；本agent不操作真实账和运行现场。

## 根移交的B45启动事实（进行中）

根确认最终KS13efd/AKA88d773上的四个匹配包、help、composition boot、正确配置unknown-stop probe、CLI认证与原LegacyExecutor L20D环境预检均通过，模型/GPU0。环境support=supported、reasons=[]，identity=null/availability=unknown保持，没有为了启动编造身份。B46仅按根消息记录，未进入B45运行环境或读取认证。

B45已注册 `schema-b45-finite-native` 新request，并正式启动唯一公开 `api.iterate`，原group自动begin/end；当前 **in_progress**。候选、writer keep/revise、第二native是否发生、实际model/job/tokens/USD/active和最终结算均待原报告，未提前通过。KS13已推送；根报告AKA88推送两次60秒超时，第三次异步重试在途，当前没有宣称远端已成功。
