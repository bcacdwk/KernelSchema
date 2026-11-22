# B17：L20D 同 parent 的较长期限模型验证

2026-11-21 UTC。本阶段接续用户“继续完成计划”与新 GPU 选择 L20D，已执行并收口一个新的600秒 native scope。**新增真实模型 calls1（coder1/reviewer0）/ GPU jobs0**；原 CLI 约52秒后以 `error_max_budget_usd` 失败，宿主 active57676ms，未超时、没有 qualification。原进程及操作已终止，原 parent 无在途操作；没有重试、业务 fixed/dynamic 或 KS writing。CLI 原始报告 USD **0.758005**；token 字段内部矛盾，真实总量保留 unknown，详见末节。

生产源码唯一可写仓为 `/home/shaohanyong.shy/atrex-kernel-agent`，起点 HEAD `804e26f79d094250a5852ad2b8a6f652f03f2928`、工作树干净。原 main 仅写本报告，其余准备文件位于 `/tmp/schema-b17-model-execution-20261121`。未修改 KS 源码、未提交/推送/派生代理，未访问、枚举或执行禁止访问的实验仓。

## 新 scope 的准备与授权边界

- 已读适用 AGENTS、B14 模型/Agate、B15、B13，以及 AKA source-release / native-probe / dynamic 和固定 KS runtime/WRITING 相关协议。
- 原准备脚本 `/tmp/schema-b14-model-real/prepare-native-scope.py` 保持不变，SHA256 `a268c6cec3ce2fb67ab7357c89dcaf081fbee4f8817d8ea61430ff06ca671d00`。
- 新副本 `/tmp/schema-b17-model-execution-20261121/prepare-native-scope.py` 将旧未创建预算提议断言改为已首次冻结的 **24 model_calls / 16 evaluator_jobs / 10800000ms active / tokens=null**，并核对角色上限 planner6/coder8/reviewer16/extract0 及原 account ID `5b6979f18703468aaa5cf9c052776d4e`。只读 authority descriptor，不打开活动时钟、注册请求或创建账户。
- 唯一原 parent bundle `/tmp/schema-b14-agate-20261121-twmkup4i`，authority `.runtime/owner.json`；Ref path `artifacts/d60b4fa1c8854d0187afefdb6368bf39/parent.json`、SHA256 `85f19bfe92d717c94950442d3fbab16e63bccf52a0f4cb8594a71c6fd02ca3a7`。不改原 H20 失败 Q 或其已终诊断。
- 新 profile 沿用 `/tmp/schema-b14-model-real/claude-code-2.1.316-qwen3.8-max-no-tools-600s.b14.json`；JCS SHA256 `f86190b7c579afe95fede90ecc762670297293b283249db6d3d181f5533cb0f0`。脚本显式核对此摘要、600000ms、USD0.50、CLI executable SHA。官方 inspect/auth 已由前阶段完成，本项不重复发模型调用。
- 将从明确固定 L20D 业务 Q 的选定闭包派生新运输 Q，只替换授权 `workflow_objective` 与新的运输预算/profile 外层配置。Task/R/Scope/Rules/seed、其余材料正文和 exact Claude profile 保留；原业务 Q 保持不变。完整七字段 `ks.claude.output.v1` 必须由真实模型返回，不注入伪结果。
- 运输目标显式要求 coder 返回完整 current_source 原文、reviewer 返回运输限定 review；不要求实现/性能/Task/fidelity 结论。新输入中的 fixed_assignment 和全部正文/alias SHA、size 将逐项核对。预览输入只记 preview，不冒称实际 capture 投递。
- scope ceiling 600000ms；新 Q 和 native workflow limits 为 coder1/reviewer1、model_calls2/jobs0/active1250000ms/tokens=null，`minimum_active_ms` coder/reviewer 各600000、evaluator1000。coder admission 保留完整 reviewer 窗口，首失败即停止该 scope；额度不足拒绝，不换账户或削减最小收尾容量。
- 后续实际调用只通过公开 `aka.workflow.schema.native_probe --scope-timeout-ms 600000`，使用原 `KS_BUDGET_AUTHORITY`。ceiling 入 anchor 后不修改；continue 不传覆盖值、不重试已付费调用。缺 terminal/tokens/USD 保留 unknown；原进程尚未确认终止则保留 fence，只结算原操作。
- 成功后 `native_probe.select_request` 的输入必须是原 L20D 业务 fixed Q；运输 Q 不作为业务授权。若尚无真实 K6 选择，不冒称完整 GPU fixed/dynamic 已可运行。

## 已完成命令

```sh
/tmp/schema-b05-a3-venv/bin/python -I -B /tmp/schema-b17-model-execution-20261121/prepare-native-scope.py --help
```

结果 exit0，仅导入与参数帮助。新副本 SHA256 `c446f5a7101ca4c9fdaca60d7f0d638f372ee2d05c84d36a062ae9331d042751`。此处初始检查尚未对任何业务 Task 执行准备脚本，也没有 core run、owner registration、reservation、capture 或收费操作；随后实际执行记录见末节。

## 初始等待条件与历史账目

B16 正在同 parent 准备 L20D 目标并持有 GPU 收费 slot；本项已交换固定材料交接需求，未读取其活跃源码。等待最终固定 KS SHA 后只更新必要 AKA gitlink/release 与来源核对，不反复重建四包或重跑完整155项。

历史全会话实际模型 calls7，其中3个120秒 timeout。已知部分15291 tokens / CLI USD `0.232672750000000010`，但整体 tokens/USD仍 unknown。原 H20 入队前400 `dev_cluster_disabled`，jobs0/models0/active531ms、原操作终止且 fence 空；本报告不把该拒绝当作 L20D 的结果，也不复写其历史原件。

## 固定来源接线的定点检查

主代理先交固定 KS `95e6df5af2f634d559208928ffad63804db99d5f` 后，从本地 Git 对象库按该精确 SHA fetch 到 AKA 可选子模块并 detached checkout；没有从 KS 活跃工作树安装。原 `tools/freeze_schema_release.py` 生成299个选定文件的 manifest，SHA256 `222443d6f130ec804803ec2fe77368805e3bbf96b676515e7633b2d4d289897d`，只同步 gitlink、release.json 和 source-release.md 三处。随后 B16 确认实测设备名含尾部空格，目标资料还需主代理最后固定；因此此处暂不提交、不据中间 SHA 启动 capture。

原环境 `/tmp/schema-b05-a3-venv` 的 isolated fresh process 来源核验通过：Schema 为本 AKA editable，另三包为原 installed wheel，字节与对应固定 AKA 源码一致；KS module/resource 同为本 AKA 固定子模块。没有重装或重建包。记录 `/tmp/schema-b17-model-execution-20261121/installed-provenance.json`。

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/schema-b05-a3-venv/bin/python -B -m pytest \
  tests/test_schema_native_limits.py::test_probe_real_synthetic_child_retains_scope_locator_digest \
  tests/test_schema_optional_module.py::test_fresh_child_installed_metadata_and_resource_source_pair \
  tests/test_schema_optional_module.py::test_stdin_exec_preserves_exact_payload_in_real_child \
  -q -x --tb=short --basetemp=/tmp/schema-b17-model-execution-20261121/installed-child
```

实际 **3 passed / 9.65s**，日志 `installed-child.log`。覆盖真实 fresh subprocess 的原 stdin/locator 与安装来源，不调用模型或 GPU。`git diff --check` exit0。最后目标文档若仅追加资源，将只更新 manifest 与复核来源，不重复该已通过组。

另准备只读 `audit-capture.py` 并完成 `--help` 导入检查；实际完成后它将复核原输入/材料 hash、no-tools init、原 exec/locator、process termination、原 CLI token/USD 和输出限定语义，不调用 owner 方法、模型或 GPU。

## 最终固定来源、输入与调用前检查

B16 实际 L20D 诊断终止后，主代理交最终 KS **`7d5fb60cf1d6f9ef7cf63dff757250bb4f614ab1`**，并明确把唯一收费 slot 移交本项。该修订保留实测 `device.name` 的8个尾空格；没有放宽 target 匹配。AKA 同步最终 gitlink、release 与说明，manifest SHA256 **`0ac0141bac462dfaafa07c66bb44d3db1b215c8a12f6d9e73e75eb99a3f34a00`**，299个选定文件；只读 isolated 来源核验再次通过，不重复前述3项。源码在调用前冻结交根；主代理精确提交为 AKA **`89fb61b696a93a255f44668658f79bd8bd0e6ce3`**，本代理未提交。最终工作树干净。

- 源码快照 `/tmp/schema-b17-model-execution-20261121/source-snapshot.json`，SHA256 `31a1eda13c29729ccf9547f30f7060ea2d1991c57c4df9a81f453c00ececb0c3`。
- 三处 owned patch `/tmp/schema-b17-model-execution-20261121/owned.patch`，SHA256 `0c5028f886fbcbb850216f9e4f7a307a198d59cd8b3848a8205684c47a50666e`。
- 最终实际 import/resource 记录 `installed-provenance-final.json`。Schema 是本 AKA editable，其余三个 AKA 包是原 wheel；KS 为本 AKA detached optional 子模块，不是 KS 活跃工作树。
- 原业务 Q：`/tmp/schema-b16-l20d-20261121-n5tg6rp7/standalone-demo/fixed.request.json`，SHA256 `8ebbdebfb3606945e528ad7ae26a66b082f9c5fab4f335de8ed4ecea638c8e1f`；Task SHA256 `4162e75b3f340e13dbe27714c17ad575f37823afc23096b473ab4e1d46bb48c2`，R0 Schema SHA256 `285960e80e5170b9ee5f3c77c93f849ffdcbe93cea42c49fedba3e60c7b62a97`。这是原 `Demo.create` 同 parent 的实际准备产物，remote=null，明确没有 K6 资格。
- 新运输 Q：`/tmp/schema-b17-model-execution-20261121/prepared/request.json`，SHA256 **`7aae71b45690950aefd03156373da0dce79c0073a978c397d2ee13ac0f9fb7fe`**。原 Task/R/Scope/Rules/seed/mode 与 exact Claude profile 全部保留，原业务 Q 未改。预检见 `prepared/preflight.json`。
- coder/reviewer 预览输入416497 / 417344 bytes，分别27正文+13/15直接alias、29个受信host implementation标识。每项正文/alias均按原hash和size核对，完整授权objective进入fixed_assignment，未削减材料。
- 实际调用前的只读 authority 快照 `parent-before-native.json`：恰2个既有已终operation，active65012ms、active_run/active_since=null、paused=false；剩余24model/15jobs/10734988ms，足以容纳新Q与完整reviewer最小窗口。准备本身未新增owner、operation或费用。实际准入继续由原 owner 执行。

```sh
/tmp/schema-b05-a3-venv/bin/python -I -B /tmp/schema-b17-model-execution-20261121/prepare-native-scope.py \
  --base-request /tmp/schema-b16-l20d-20261121-n5tg6rp7/standalone-demo/fixed.request.json \
  --authority /tmp/schema-b14-agate-20261121-twmkup4i/.runtime/owner.json \
  --out /tmp/schema-b17-model-execution-20261121/prepared

KS_BUDGET_AUTHORITY=/tmp/schema-b14-agate-20261121-twmkup4i/.runtime/owner.json \
  /tmp/schema-b05-a3-venv/bin/python -I -B -m aka.workflow.schema.native_probe \
  --request /tmp/schema-b17-model-execution-20261121/prepared/request.json \
  --out /tmp/schema-b17-model-execution-20261121/native-capture \
  --scope-timeout-ms 600000
```

## 真实终态、费用矛盾与停止点

公开 probe 进程 exit0；domain result **blocked**，reason `original coder probe failed: native_terminal_unavailable`、qualification=null、synthetic=false、recovery=none、unsettled=[]。原 CLI 子进程 exit1、timed_out=false、result subtype `error_max_budget_usd`、terminal_reason `budget_exhausted`、errors `Reached maximum budget ($0.5)`。CLI duration51519ms，宿主本请求 active57676ms；不是600秒超时或scope期限耗尽。

| 原身份/事实 | 实际值 |
| --- | --- |
| Run | `90bc0a6376cc4488ae09e1d4ac5cad0f` |
| 原 coder operation | `aka_b52d235c2d7e463a84bdcc2a6b693ddf` |
| 原 session | `bb595a61-adb9-4d0c-8af6-c5c480daf945` |
| 原 PID / start_ticks | `1095032` / `162645832`，已确认不存活 |
| 实际输入 | 416497 bytes；SHA256 `b7bc1cc2023c93efbae38ffc510d7e6a456d5ddc072d911c73fefe033ffd973f` |
| assignment SHA256 | `f2e718a5760188486daead01eb7ad1732d4c7a9566334c5ab814da5148487f4b` |
| 原 invocation Ref | `artifacts/57fc89a2d039483cb080ce30dcc2f01f/native-invocation.json`；SHA256 `1ffd2052a3558613e16f16f5fa836bea8c460be20651e72ddd8ed0644b1081e5` |
| usage index Ref | `artifacts/252cb46f9b7c9618bca9e67b7d80ac10/usage.json`；SHA256 `190bb160d29c4026b92c66ff20e4c262d0915676fe14868e9b2a454c80acdb4f` |

真实 init 为 `qwen3.8-max[1m]`、permissionMode=dontAsk、tools/MCP/skills均空、仅原profile列明的两个builtin plugins。原exec记录与locator摘要、session/operation/assignment/input一致，scope及operation ceiling均600000ms。实际输入与preview的字节摘要不同，因为实际capture新发布了同正文delivery的Ref；授权objective、assignment、每项正文/hash/size均复核一致，不用preview摘要冒充实际投递。

CLI 原 terminal 的 `total_cost_usd` 与 `modelUsage.costUSD` 一致为 **0.758005**。本profile保持原0.50阈值，CLI在超过阈值后停止；该参数从未承诺宿主硬美元上限。没有人为调高阈值或重试。terminal aggregate usage 的四项token全0，而同一个原 terminal 的 modelUsage 报 input6/output2219/cacheCreation112400/cacheRead0，组件合计 **114625**；assistant事件另重复报input112354，不与terminal重复相加。

当前 AKA 原parser采用aggregate，故原receipt/owner投影保留 `tokens=0, measurement=exact`。这是原件字段之间的矛盾，不能解释为免费调用或已知真实总token；**本项真实总tokens记unknown**，并列保留所有原字段，未改账、未回填用量、未替换receipt。只读诊断 `terminal-usage-diagnostic.json` 记录上述差异，`capture-audit.json` 记录原状态。原 assistant 唯一text严格符合RoleOutput且content精确等于current_source；但原terminal失败，所以这段text只作诊断，不能拼成成功terminal或transport qualification。

首失败后scope phase=done，reviewer0，未调用continue、未重试、未生成selection、未启动业务fixed/dynamic或KS writing。所有原材料、失败输出、profile、authority、source binding与原调用原件保留。

最终共享parent：总model_calls1（coder1）、evaluator_jobs1（B16）、active122688ms；三operation都已终，active_run/active_since=null，unsettled=[]。剩余23model/15jobs/10677312ms；本项不把余额当作自动重试许可，收费slot已交回主代理。B16 worker缺bwrap，namespace/readonly及六作业GPU资格仍未通过。

全会话实际模型累计 **8**；三个旧120秒timeout仍为原unknown费用。本项新增CLI已知费用0.758005，与历史已知部分合计CLI USD **0.990677750000000010**，整体费用与tokens仍unknown，不把已知部分当总账。

完整交接 `/tmp/schema-b17-model-execution-20261121/b17-final-handoff.json`，SHA256 **`d49e7603d1635ea9f0c92b33b6950f36fe228ec9f82d6cab8684d3be71f83528`**。本轮没有修改native生产逻辑，也没有修改旧scope、旧失败账户或原H20/L20D业务Q。
