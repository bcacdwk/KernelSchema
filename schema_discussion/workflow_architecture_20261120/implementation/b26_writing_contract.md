# B26：写作输入合同补全（CPU）

2026-11-21 UTC。基于 KS worktree `e5a0748b81498cdecd97e12986491e250044b204`，补全 B24 真实 design 暴露的写作输入缺项。生产改动限于 `kernelschema/writing.py` 与 `kernelschema/_writing_policy.py`，另更新 WRITING 说明、定向测试和一份逐字保留的 B24 失败稿测试 fixture。**新增真实模型调用0、GPU jobs0、真实 owner 操作0；没有重试 B24，也没有把旧失败解释为成功。**

固定文件清单见 [source-manifest.json](/tmp/schema-b26-writing-contract-20261121/source-manifest.json)，SHA `c779b0c6f3b3df846c8f34a603b7d49cec09f72dc71e7a0d6538617ec23c88ba`；五文件逐字副本在 [frozen-patch](/tmp/schema-b26-writing-contract-20261121/frozen-patch)。本项没有 commit/push，待根统一固定源码与同步下游。原 main 仅写本报告。

| 改动文件 | SHA-256 |
| --- | --- |
| `kernelschema/writing.py` | `6abb9f9e9c039e8227af2fcba06dea21d5d5036af4e58bebfce810c48cff869c` |
| `kernelschema/_writing_policy.py` | `c94969aff8268707f3b1af05f62f888ddae57a15fb92ca46024d90b1d18fa024` |
| `kernelschema/WRITING.md` | `c19a776c544d0a68429ab326e4922a1e2350b55161cdeb27866864f4853b182b` |
| `tests/workflows/test_writing.py` | `e59607a852e3da981de2d1e98992df65f636010fdb66deb328c00a30b4988211` |
| `tests/workflows/fixtures/writing/b24-failed-draft.json` | `66f919efa8c676cd8f241cf82563c34c41c3d8e020f5c0c1b5c9f3e5596e2892` |

`delivery/writing_input` 新增 `selection_catalog` 与 `text_policy`。前者直接从当前 Materials 的 planner grants 生成，每项保留真实 bare `id`、不可变 `ref`、`delivery_id=material/<id>`，并明确可用于动作的 `action_id=material:<id>`；只有同时获 planner/reviewer 授权的 material 才有非空 action_id。selection 继续只接受原 bare ID，action_inputs/action_evidence 继续只接受原 action_catalog 的精确键，未扩大权限。Reviewer 收到相同 text_policy，不额外收到 planner-only selection catalog。

有限 METHOD 直接说明三个 ID 命名空间、全部 draft 字段、`bindings` 必须是 `{coverage_ids,node,variant}` 项组成的列表、design 必须 `revise + next_action=none` 及 null/空动作字段、只有 optimize 可提 implement/verify、keep 的 null/空形状。实际字段集合是 format、decision、reason、selection、raw、dispatch、bindings、next_action、action_reason、action_inputs、action_evidence。新 METHOD SHA 为 `bc90e3c710b867acc22ee3c9f248ed79e8e19e352aee7483cd51d8ec8bc76349`。`parse_draft` **整个函数与 e5 逐字相同**，函数源码 SHA `42a992de7f197b9fd770f795464f40cd5b3051543a70eb58c6c67c7ad101de13`；未添加 strip、截断、对象转列表或动作自动改写。

`text_policy` 从实际原 Template 模块的 **80条 BUDGETS** 直接投影 `path/maxLength/maxLines/prefix`，并携带该冻结规则的真实 Ref/hash 与 `BUDGETS/text_size/check_values` 符号来源。原 `schema_template/tools/text_policy.py` SHA 仍为 `3bb21e99a3872f94a00c296bbe52b18395c2f9a9fc05c13a00221ccb95ee69c0`。包含既有 Thesis 300字符/2行、Strategy 与 verb describe 150字符/1行、meaning 50字符/1行等全部 BUDGETS 项，没有手写第二套 Template 验证器。

计数规则也显式投递：先按 YAML 折叠/转义得到原始字符串值，再使用 `len(value)` 计 Unicode code points（包含空白和换行），使用 `len(value.splitlines())` 计逻辑行（包含内部空行，末尾终止符不另增一行）；不 trim、不做 Unicode normalization。`*`/`[]` 分别表示路径上的命名映射项/列表项；非空 prefix 仅匹配对应前缀，但计完整值；maxLength=null 仅表示该项不设字符上限。模型必须在返回前自检，原 Template 仍执行全部结构、引用与其他字段规则。

当前 runtime 没有独立 response JSON Schema 参数，因此只新增 writing 专用的显式声明；没有改通用 CLI 参数、K3 外层 envelope、native profile 或公共投递机制，也没有声称 JSON Schema 能验证 YAML 字符串内部全部语义。Template、Task/acceptance、原 Schema、evaluation、budget、`_stage_ro`、D3 默认、美元/时间上限及单 owner 规则均未改。没有访问禁止的实验仓、派生代理、安装依赖或调用真实 CLI/认证/GPU 服务。

对 B24 实际 Materials 的 [CPU 审计](/tmp/schema-b26-writing-contract-20261121/audit.json) 使用独立 [审计脚本](/tmp/schema-b26-writing-contract-20261121/audit.py)：只读原 Q/材料并在自己的 `/tmp` 发布一个明确未准入的预览 Q，冻结完整 standalone Rules（保留 fidelity METHOD），经现行 `_prepare` 和 `prepare_role_input` 得到 [planner-preview.json](/tmp/schema-b26-writing-contract-20261121/planner-preview.json)。该预览没有 owner、intent、operation、result 或调用收据，不是一次实际模型投递。

| 预览检查 | 结果 |
| --- | --- |
| 真实 planner selection_catalog | 20项，均与实际 Materials、正文 Ref、delivery ID、可用 action ID 一致 |
| text_policy 大小 | 10263 bytes，80条，与原 B24 Template source hash 相同 |
| 新 writing input | 22386 bytes；新增2字段，原5字段保留 |
| 完整 planner payload | 432947 bytes；原 B24 actual 为414153 bytes |
| 材料投递 | 69项：24正文、16alias、29 host-only，与原 B24 数量相同 |
| payload SHA | `84399d915d01304593876227019e89775225aa3ba237e8856a905e74d169b6dd` |
| 授权边界 | 无私有 evaluator、R0 Design、current_source；未恢复宿主实现正文 |

预览复用原 Task/scope/Materials/profile/budget 的完整公开 Ref；只为 CPU 审计生成新 request_id 与新 Rules，不打开真实 authority。未准入预览的数字不是新实际 run 的费用、用量或资格。

[原正文保留补审](/tmp/schema-b26-writing-contract-20261121/semantic-body-preservation.json) 逐项核对原 B24 的正文/alias：除明确替换的 writing METHOD 与 writing_input 外，其余38项原正文全部逐字保留，包括 standalone fidelity METHOD；没有因摘要投递而删除原语义文档。

原 B24 retained draft SHA `66f919ef…`、raw SHA `8a440621…` 原字节均保持，生产结论仍是 **failed/new_request，Schema=null，未到生产 precheck/review**。其555.801秒、142547 tokens、原 CLI USD1.44354875 作为原历史保留，本项没有修改原账本、failed Q/profile 或旧结果。新版 parser 在同一19个 `material:` selection 上仍报原错误。对同一未改 raw 单独调用原 `validate_documents` 仍得到8处长度错误：Thesis 720/469/522>300，Strategy318>150，load/store describe165/183>150，Partition/Storage meaning51/69>50。此静态检查不是旧 run 的生产 precheck，也不反填旧结果。

另外从原稿静态可见 `bindings` 是 Scope 对象，且初始 design 提了 `next_action=implement`；二者与原 parser 既有合同不符。已在 METHOD/WRITING 中明确，未先去前缀、改列表或改动作再把原稿送入成功路径。新正常 design/optimize/reviewer 验证使用独立的、标明 SYNTHETIC 的正确裸 ID/短文本 fixture，与旧真实失败稿严格分开。

定向验证命令：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B \
  -m pytest -q tests/workflows/test_writing.py \
  --junitxml=/tmp/schema-b26-writing-contract-20261121/writing-tests.xml
```

结果为 **71 passed，460.92秒，退出码0**。完整记录见 [writing-tests.log](/tmp/schema-b26-writing-contract-20261121/writing-tests.log) 与 [writing-tests.xml](/tmp/schema-b26-writing-contract-20261121/writing-tests.xml)。检查涵盖真实 fake-process/API design、optimize keep/revise、reviewer、完整字段预算/材料映射、裸 ID 与动作 ID 区别、未授权/私有材料拒绝、原稿无变换拒绝、同 parent 预算与恢复/重放。`git diff --check` 通过，冻结文件逐字复核通过；没有全库重跑。初次 pytest 自动插件发现因环境缺失 `platformdirs.pytest_plugin` 未加载测试，随后显式禁用第三方插件自动加载；首轮新增用例13通过、1个测试误用只生成计划的 `normalize_directory`，已把该测试改为原 `validate_documents`，没有为此修改生产逻辑。

**源码/Rules 耦合：**新 METHOD、writing.py 与 policy.py 均被 `_rules` 逐字检查及 backend.identity SHA 绑定。以 e5 冻结的旧 Demo writing Rules 不能直接换新实现消费；应在首次真实资格作业之前从根固定的新源码重建尚未准入的完整 Demo/Q。此事实已同时告知根与 B27；没有放宽 identity 或引入迁移平台。通用 runtime/profile 未改，不意味着旧 writing Rules 自动兼容。

后续真实修正只建议在根固定源码后另开明确新请求：在新的 workflow_objective 中写明“修正随附的失败提案，保持原 Task/coverage/硬判据”，将逐字原 retained draft 与逐项失败诊断发布成 planner/reviewer 获准 Materials，例如 `b24_failed_draft`、`b24_failure_diagnosis`。标注它们是未通过的原提案/静态诊断，不是 accepted Schema 或测量证据；新的 selection_catalog 会自动加入其真实裸 ID。要求模型按新 envelope/ID/分支规则重答并自行缩短文本，不能由宿主代改旧稿。新 Q、Rules、profile、输出目录与请求 ID 由新固定源码创建，继续显式连接同 B14 parent；旧 failed Q、profile、原收费和结果全部留在原处。实际准入、余额核对与任何真实模型调用由后续根授权阶段办理，本项没有执行。
