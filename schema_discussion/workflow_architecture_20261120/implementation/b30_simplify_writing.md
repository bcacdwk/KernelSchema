# B30：design/optimize 默认只写 Schema，标准 AKA 衔接

2026-11-21 UTC。按用户最新“能跑即可、自定义能减则减；design/optimize 只写、implement 参考 AKA”的指令，在 KS worktree `schema-workflows-v1`、基线 `7492611567d6a5bdd5983bb98126001d0b15e5e8` 完成本阶段改动。没有 commit/push；没有访问禁止的实验仓、派生子代理、修改原 extract/Template 规则、旧评测原件或真实 B14 authority。**真实模型调用0，GPU jobs0；本报告中的运行结果都是明确 synthetic 的 CPU 工程验证。** 原 B24/B29 失败、收费与终结状态没有改写。

## 交付与边界

默认 `WritingBackend` 为 `write_only`：planner 一次、reviewer0、evaluator jobs0。内层输出只需 `raw:[{node,text}]` 和简短 `reason`；optimize 可只返回 `decision=keep` 与 `reason`。旧 selection/action catalog、锁定 Contract 重复包装、11字段 envelope、80项文字预算、Template resolve/precheck、派生 Scope 语义和独立 reviewer 四门不再是默认发布前提。旧 strict 路径保留为 `writing_mode="strict"` 显式选项；没有给旧失败补“成功”状态。

实际产物仍使用现成 Schema/Scope Ref、Publication、ClaudeRuntime、原生回执和 OwnerAccount，没有复制 AKA 平台或建第二套账本。`raw` YAML 原样保存，`change_report`/`rationale.md` 说明原因与不确定性，`extensions.information` 给出 `schema_directory`、`candidate_draft`、`retained_files`、`call_artifacts`、diagnostics 和未评估声明。Schema 原始目录可直接供下游 coder/AKA 读取。结果导出显式带这些原件；不靠任意字典递归猜依赖。

仅安全文件名、唯一 root 节点、UTF-8、非空 YAML mapping 属于基本文件可读性约束。Template 长度与语义不挡发布。坏 YAML 仍保存实际 Schema 原文并返回 partial+诊断；缺文件/非法路径保留原始 JSON 和逐份原文本，不静默清空。不会宿主缩短正文、规范化旧失败稿或编造性能收益。写作 completed 只表示文件写出；permissions/semantics=not_run，structure/publication 只关联实际文件与可读性记录。

存在于 artifact closure 的旧隐式 Template.resolve 也已收敛：显式标记 `Schema.extensions.information.writing_mode=write_only` 时只核文件身份/安全路径/Ref闭包。其余旧 Schema、原 extract 与 legacy implement 仍走原检查。这是兼容现成 Ref 的窄变更，不建立平行 artifact 系统。

## 简洁入口与材料

新增文件 API `api.design(task=..., out=..., references=[...])`，`api.optimize(task=..., schema=..., feedback=[...], out=...)`，CLI 同义。任务、目标、环境、明确参考、当前 Schema 与反馈均按实际文件完整复制。为免伪造评测合同，Task 的 evaluator/baseline_source 可为 null，**只允许 write_only 消费**；legacy implement/strict 请求继续要求真实评测 Task。默认没有 readonly evaluator 资格要求。

可复制命令：

```sh
kernelschema design --task /materials/task.md --out /runs/design \
  --reference /materials/neighbor.yaml --runtime-profile /materials/claude-profile.json
kernelschema optimize --task /materials/task.md --schema /materials/current-schema \
  --feedback /materials/actual-feedback.txt --out /runs/optimize \
  --runtime-profile /materials/claude-profile.json
kernelschema implement --schema /actual/written/schema-directory \
  --problem /bench/problem --repo-root /checkouts/atrex-kernel-agent \
  --platform L20D --max-iters 2 --out /runs/implementation
```

CLI 打印实际 Schema 目录。`--request`/`--continue` 保留；继续原操作不重开调用。`runtime_profile` 接受 ClaudeProfile、字典或JSON文件。未给 profile 的简洁入口只探测已安装 CLI，然后用 `qwen3.8-max[1m]`、high effort、600秒、2.00 USD 原生停止阈值，现有 CLI auth；不是硬账单上限，也没有自动 retry。显式传入原 profile 则保留全部原参数。high 是减少纯写作耗时的合理默认，**没有以真实调用证实它一定完成**；后续根的真实验证仍应先固定代码/显式 profile。

默认 Rules 只选择写作 METHOD、当前 operation 的 DESIGN_SCHEMA/OPTIMIZE_SCHEMA、common 与 starter YAML；显式 references 可补需要的专门 DOC/wiki/近邻，全部所选正文不截断。standalone 的写作 Materials 不再额外 dump 全 DOC0–6、完整实现方法、Template checker 或 fidelity/read-only 合同；实现材料由 AKA 标准流程负责。

[CPU 输入预览](/tmp/schema-b30-simplify-writing-20261121/input-size.json) 为默认 row-softmax design：80,637 bytes，21 entries（15正文/6alias/0 host），owner operations={}。预览未准入模型/评测。B29 的463,176 bytes还含旧失败参考，是不同输入，不把二者声称为等价材料基准或真实用量差额。预览脚本与完整 payload 在同目录保留。

## 标准 AKA 与有限自动迭代

KS `api.implement(schema, problem, repo_root, workspace, **native_options)`/CLI 直接转发 B31 的同步 `aka.workflow.schema.standard.implement`。不走旧 reference 的 mandatory fidelity/readonly gate；AKA 原 Light/Agent/Problem/Executor/Evaluator 负责实际代码与评测。platform 是显式硬件 alias（如L20D），CLI不会把cuda当成可用默认设备。结果原 JSON（含float）不翻译成虚构的 KS 五项通过。

standalone 默认止于写作；显式 `--aka-config` 后可 design→standard implement→一次 optimize→若 revise 再一次 standard implement。`--start optimize --schema ...`、fixed/native 开始也可选。driver 将原 AKA 返回JSON与 B31 声明的 `feedback_files` 原文传给 optimize，并将写作 rationale 交给 AKA docs；不新增评测器。原始 native workspace 路径和结果保留。完成写作不等于该次设计效果好。

```json
{"problem":"/bench/row-softmax","repo_root":"/checkouts/atrex-kernel-agent","platform":"L20D","framework":"Triton","max_iters":2}
```

默认 AKA 使用原生预算/结果记录。只有显式 `share_owner:true` 才传 driver 原authority；需要的 `owner_limits` 原样转交 B31 可选薄适配。并未读取/操作真实B14 authority，也未给旧scope续额。没有保存结果的中断 native campaign 会明确停在原workspace待结算，不自动重跑可能已收费的调用；这是当前有限恢复边界，未冒称通用恢复平台。

## 验证

统一解释器为现有 `.worktrees/.tooling/k1-evaluation/bin/python`，`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，无安装/下载。各组日志、JUnit 保留在 [/tmp/schema-b30-simplify-writing-20261121](/tmp/schema-b30-simplify-writing-20261121)：

- `tests.log/tests.xml`：60 passed / 43.39秒，write_only + contracts + artifacts + API。
- `compat-tests.log/compat-tests.xml`：新写作/CLI/export及 strict design/optimize 14 passed；1个strict driver失败来自未提交的中间Rules Ref。已修复为strict模式原完整Rules，`strict-driver-tests.log/xml` 单项重跑通过（33.58秒）。
- `extract-tests.log/xml`：3 passed / 15.36秒，原Template parser、不改原件、extract原CLI透传/退出码、原receipt闭包。
- `entry-tests.log/xml`：3 passed / 5.54秒，后续入口/默认driver窄变更。
- `platform-tests.log/xml`：最后明确platform参数后，standard API/CLI透传单项通过。

这些组存在重复；合计覆盖68个不同测试用例，所有失败用例在修复后通过。没有全库重跑。检查覆盖长正文不丢弃、无锁Contract/Template/reviewer执行、坏YAML/路径保留原稿、optimize keep/revise和原反馈、无evaluator文件API、真实fake子进程单次用量、默认driver零GPU、native mock两次实现之间一次optimize、原件导出、原run重放。后续真实模型/GPU结果仍未建立。`git diff --check`通过。

## 源码清单

[manifest](/tmp/schema-b30-simplify-writing-20261121/source-manifest.json)，SHA `0eb28992a406624fce9d45c9a163cd1bf57da1d9dadc2ebe4d51b061f0609f1a`；20个改动文件逐字副本在 [frozen-patch](/tmp/schema-b30-simplify-writing-20261121/frozen-patch)。只有这些KS文件归本agent写入；AKA实现由B31负责。根可按此精确提交，未产生本agent commit/push。

| 文件 | SHA-256 |
| --- | --- |
| `README.md` | `cd27b433d6eb55867f840c3bc94750cd9650619cf78f3f715dfb5b4fb7bd97eb` |
| `docs/workflows/protocol.md` | `15b24710ee216cde19ae5a49582aa427baf180f8e28c0204070adde5d61a8fda` |
| `docs/workflows/standalone.md` | `e91f7330bdfaf9535f8ed4b46e6f2e9d795bad366c2a1dc59773a8911baaf62e` |
| `examples/workflows/row_softmax/driver.py` | `e10c5a85edef2679b40298a83a108f5c9af30a65ab7f5f43c9f65d88a8fe0eee` |
| `examples/workflows/row_softmax/run_demo.py` | `55ab31ca3b3a68cfbfbdacdff1285ee43d32bef27ee0cd8966bfdee62e8f66e4` |
| `kernelschema/INTERFACE.md` | `23a945b163758339dc0f3fa21bacad3608c82ec458a28f9290e12cbeeaacb550` |
| `kernelschema/WRITING.md` | `39f6dc712904c63a7319e8215614974f76714ae2a5be588ed96db828a57254c7` |
| `kernelschema/_writing_policy.py` | `917706a3f35c127666e59a9042b3a6e0364850fa2afa8f97be980d4e4e35b842` |
| `kernelschema/api.py` | `da20165e48676c1c0ba96c2c79849b79d72203c99e82fecca06fb173547c2e19` |
| `kernelschema/artifacts.py` | `23fb7969048f7a1c180b65552df94efae0013679ca2eddc8ee9e6a2a2e31166b` |
| `kernelschema/cli.py` | `b61a0319674726456ac7cdc785b3f6f0f28f718dc8182334cbef2426f7584919` |
| `kernelschema/contracts.py` | `8015a58187781372b6e015dbdf6a8024ed88d1d0e4e5fa88c9b0d4df26f40e3c` |
| `kernelschema/materials.py` | `632151bc9514705de96bbb88afeadfdb44e0c8ab8e987dbe09bb0b413fc3ebe9` |
| `kernelschema/runtime/claude.py` | `9c6570051d943f96c3272c96b5d5c3470c338ee1e355971d384511e5692ae19f` |
| `kernelschema/writing.py` | `1c1ab9d7f670ec99c9e7ebd3cb2972d6a5a9a687054804d3918575b58ba0e2c5` |
| `schema_design/README.md` | `e7f785aab59e1aaaae2aa582a0c7fc801a3bf548f6ef8a9d9cb857d7bdf8b688` |
| `schema_optimize/README.md` | `e879a71fe390aa1dadec2db3793c061b342ad044b00876e96c937ab478d58b6a` |
| `tests/workflows/test_standalone.py` | `966058df9f9182653ba358e2338ab27b476e99cbd6005cb377ed55f4bcb7f2bf` |
| `tests/workflows/test_write_only.py` | `f918a428c7efb9ac990a8fb0e21e7eb015767a4d27bb836f20d38a581874a0a6` |
| `tests/workflows/test_writing.py` | `e0a07188d35e08026a631b02e513bb170e3536527fe97be8b2faa66bf85ddac9` |
