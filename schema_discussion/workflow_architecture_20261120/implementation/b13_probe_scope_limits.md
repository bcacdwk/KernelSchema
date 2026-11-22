# B13：native qualification scope 的显式不可变期限

本修复仅为新 AKA native qualification scope 增加 `--scope-timeout-ms`。默认 120000 ms，声明 profile 更短时冻结为该较短值；显式值必须是正 safe integer 且不超过原声明 `ClaudeProfile.timeout_ms`。原 Task、request/profile 身份、OwnerAccount、USD threshold、调用额度和实际绝对截止均不扩张。新增真实 MODEL/GPU/网络作业为 **0/0/0**。

## 固定边界

- 唯一源码写仓：`/home/shaohanyong.shy/atrex-kernel-agent`，分支 `feat/schema-workflows-v1`，开工固定 HEAD **`e867333dfcbbce553465635bcf57a9bde842b06f`**。本代理不提交、不推送、不派生代理；最终提交由根完成。
- KS 子模块保持 detached **`1c9e592c6e0ec021f91e17af5a8c39956ce39178`** 且干净；gitlink 和 release manifest 没有改动。manifest SHA-256 **`f85aa4e254407d319318520aab49b7a448b4d77dd7a7551baffee490d78fb14e`**。
- 已读适用 AGENTS、AKA 公开 source-release/optional-module/native 支持说明、native_probe/calls/session/Transactions/native_admission/exec 实现、B10 最终 AKA 报告和 H052–H058。未访问或运行 AKA_experiment，未访问原私有 capture/run/account/request 内容，未运行原 scope 的 continue。
- 只读核对原保全目录的指定 `INDEX.json`：SHA-256 仍为 **`01260d28efd3ed82191d48f796289e66c3ff78e25c86a45ade6bb03db21769a2`**。没有改写归档、旧 wheel 或原真实失败；本轮没有重新逐项读取该索引所列的 680 个文件。

## 实现与恢复语义

1. `native_probe.identity(profile, scope_timeout_ms=...)` 在 `prepare_run` 前校验新参数，把解析后的 `scope_timeout_ms` 写入原 run 的 backend implementation identity。`capture` 只消费这一 anchor 值，没有运行中 override 参数。参数坏值在创建新 run/owner 和调用模型前失败。
2. `Transactions.binding` 必须与该完整原 identity 相等。进入原工作前检查缺失或改变的 scope/事务绑定、scope 记录和原 operation ceiling；continue 缺 state 时拒绝重建。即使库调用者漏传 `continuing=True`，已有 scope state 也强制 `Transactions(..., continuing=True)`，原 budget binding/authority 丢失时不得重新创建或注册账户。缺 state 但仍有原 intent、budget/owner 文件、native/terminal 记录或 qualification result，也在建账前拒绝。原 paid scope 没有迁移、默认回填、续领或新账本。
3. `--continue` 在打开 run 前拒绝 `--scope-timeout-ms` 和 `--out`。它按原值重新验证实现身份，不调用当前默认来覆盖原 300000 ms。原代码身份不匹配时保留原版本进行原工作结算。
4. 原 `RoleCalls.invoke(maximum_timeout_ms=...)` 继续持久化每 operation ceiling。`RoleSession` 使用原 profile/request/caller 的最小值，再交现有 `NativeCallAuthority` 与 KS 原绝对截止 helper 按 owner/native reservation 取紧；没有新计时层。业务调用的默认 `None` 不变。
5. Locator 新增 `timeout_ceiling_ms` 表示 caller 原声明 ceiling，原 `maximum_timeout_ms` 仍表示 profile/request 收紧后的 ceiling。实际 `stdin_exec` 在原 exec 记录里加入 `locator_sha256`，把 scope ceiling 与原 session/input/request/role/实际最大值一起绑定到原进程记录。
6. Qualification artifact 增加 `scope_timeout_ms`。`native_admission` 与 `validate_capture` 验证原 exact Claude 摘要、显式 scope ceiling、locator ceiling/实际最大值和 exec 的原 locator 摘要，不再用常数 120000 验证全部资格。缺值、篡改值、换 profile、缺 exec 绑定均拒绝。更大 ceiling 只代表允许的上界，不证明实际运行满该时长；这些仍是可信本地主机原执行证据，不是签名服务证明。

公开可复制示例在 `docs/schema-source-release.md`，optional-module 链接同一语义。新 profile 至少声明 300000 ms 时可用：

```sh
python -I -m aka.workflow.schema.native_probe \
  --request "$DEMO/fixed.request.json" --out "$PRIVATE/native-qualification-300s" \
  --scope-timeout-ms 300000
python -I -m aka.workflow.schema.native_probe \
  --continue "$PRIVATE/native-qualification-300s"
```

依赖仍先安装 KS editable 与 runtime dependencies，再以 `--no-deps --force-reinstall` 安装同版 0.2.0 AKA wheels；没有更改四包版本或依赖安装规则。

## 源码变更清单

以下均为 0644；最终完整 size/hash/status 在 `/tmp/schema-b13-source-evidence-final.json`。只有这 8 个 AKA 文件改变，无 KS/gitlink/manifest 差异。首版证据 `/tmp/schema-b13-source-evidence.json` 保留，不能代替下面的最终两个增量文件 SHA。

| 相对 AKA 路径 | SHA-256 |
| --- | --- |
| `docs/schema-optional-module.md` | `e481d9587a11137cd83179ff49aee447094fae74aa027fde7154e09aed742f75` |
| `docs/schema-source-release.md` | `d047e5dc7f76ebbef5d50eda651120e7500ed42fdb846b1202b903adb510bc62` |
| `src/aka/workflow/schema/agate.py` | `f4063a9d993e5b51ac147f8fbdfa96ec863cba621e74c3b3c2817519168010b0` |
| `src/aka/workflow/schema/native_probe.py` | `dbd896b4b879d932f54bb7def6008c1b51bacb9b3b3bd38868a908593cdb5b73` |
| `src/aka/workflow/schema/session.py` | `3b7c1331997a42dbdd861d2f825455d3841999d0efb7e943858aadf03c7c1225` |
| `src/aka/workflow/schema/stdin_exec.py` | `504d4c030654f9b88f34b2d345df370d0329b546031483851ddd17c66ce67cb1` |
| `tests/test_schema_native_limits.py` | `34e597c9d4a5069ce5c0a744129d0998f808a6aceac2da956c1949823a0a6ba9` |
| `tests/test_schema_native_probe.py` | `24eab8cc34ba028e9f0ede7d2895a46a0497e4ec53ab911538565ec4423a5a69` |

## 定点验证

使用原 `/tmp/schema-b05-a3-venv`（Python 3.13.12）；开工从 `/tmp` 以 `python -I -B` 核实 Schema 来自本 AKA editable，KS module/resource 来自固定 1c9 子模块。无真实模型/GPU或网络调用。

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/schema-b05-a3-venv/bin/python -B -m pytest tests/test_schema_native_limits.py tests/test_schema_native_probe.py -q -x --tb=short --basetemp=/tmp/schema-b13-scope-cpu-final

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/schema-b05-a3-venv/bin/python -B -m pytest tests/test_schema_optional_module.py tests/test_schema_backend.py tests/test_schema_dynamic_recovery.py -q -x --tb=short -k 'fixed_role_native_first_repair_retry_and_reopen or complete_native_receipt or existing_ks_owner or fixed_request_rejects or owner_denial or native_invocation_retained or unknown_native_return or finish_recovers or V0_prefetch_group or paid_production_finish' --basetemp=/tmp/schema-b13-original-recovery
```

| 验证 | 实际结果与日志 |
| --- | --- |
| 新 scope 与原 probe 回归 | **11 passed / 70.70s**；`/tmp/schema-b13-scope-cpu-final.log` |
| 原角色、同 owner、拒绝权限/profile、保留 invocation、unknown、V0 prefetch、paid finish 恢复 | **10 passed / 51 deselected / 107.94s**；`/tmp/schema-b13-original-recovery.log` |
| 根复核后补库调用默认参数的原 owner/state 丢失，以及原 CLI 300 秒与 receipt 恢复 | **3 passed / 31.42s**；`/tmp/schema-b13-owner-gap-final.log` |
| 最终实际安装 wheel 的 fresh-child 来源、真实 synthetic 子进程 locator 摘要、精确 stdin | **3 passed / 6.99s**；`/tmp/schema-b13-installed-child-final.log` |
| Python AST、Markdown 本地链接、`git diff --check` | 通过；源码证据 JSON 记录 AST/链接，git 命令 exit 0 |

scope 组具体覆盖：默认 120 秒与 5 秒短 profile；显式 300 秒真实 RoleSession/NativeSession 到 synthetic process 的 timeout 大于 120 且不超过 300；原 owner/native 180 秒、原 request 90 秒进一步收紧；原 Task/request/profile 对象与 exact Claude 摘要不变；bool/float/string/零/负/超 profile/超 safe integer 参数零调用拒绝；CLI 中断后保留原 receipt、300 秒 continue 不回填 120、不 replay；缺 anchor cap/state/binding/probe/operation cap 拒绝；资格 cap/locator/exec 摘要篡改和换短 profile 拒绝。

另有真正 fresh subprocess，经过原 `stdin_exec` 和 NativeSession 执行只位于测试临时目录的 synthetic Python CLI，核验原 exec `locator_sha256`；它不运行真实模型，注入的 Agent 使 capture 明确标记 `synthetic=true`。全部期限断言检查传给进程的上界，没有等待 120/180/300 秒来冒充长时间真实运行。

首轮日志 `/tmp/schema-b13-scope-cpu.log` 为 **2 passed / 1 failed / 17.15s**：新增 negative fixture 把 native exec JSON 误传给仅接受 KS protocol 的 `Publication.put_json`。测试改用原 native artifact 写法 `put_bytes(jcs(...))` 后执行上述完整定点组通过；生产实现没有因该测试错误改变。

根独立复核发现公共 `capture(context)` 漏传 continuing 时，原 state 存在但 budget binding/authority 丢失可能先走 `create=True`。按同一恢复边界补 `continuing or existing` 与既有事实守卫后，只增跑以下 3 项，未重跑前面 11+10 组：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/schema-b05-a3-venv/bin/python -B -m pytest tests/test_schema_native_limits.py::test_default_capture_never_recreates_lost_original_owner_or_scope tests/test_schema_native_limits.py::test_cli_continue_preserves_long_scope_and_rejects_lost_bindings tests/test_schema_native_probe.py::test_probe_lost_return_reuses_original_receipt -q -x --tb=short --basetemp=/tmp/schema-b13-owner-gap-final
```

新增测试在已完成的 synthetic scope 上移除原 budget binding、authority、state 及组合，并禁止 `OwnerAccount.create/register`，验证默认 capture 仍拒绝、缺文件不重建、调用不增加；新 prepared scope 只残留任一种原事实时也拒绝新建。首次 `/tmp/schema-b13-owner-gap.log` 为 **1 failed / 6.91s**，实际已正确 fail closed，但测试正则没包含底层 `owner-account.json` 缺文件消息；仅修正测试消息匹配后 3 项通过。

## Schema 单包工件与实际安装

仅构建改变的 Schema 包，复用原 builder ownership 和 sdist→wheel，并以原隔离环境 `--no-isolation` 构建；未重建 Framework/Legacy/Light，未覆盖 `/tmp/schema-b10-aka-dist-final` 或旧 wheel。

```sh
/tmp/schema-b05-a3-venv/bin/python -B /tmp/schema-b10-aka-build.py --schema-only --output /tmp/schema-b13-aka-dist-final
PIP_DISABLE_PIP_VERSION_CHECK=1 /tmp/schema-b05-a3-venv/bin/python -m pip install --no-index --no-deps --force-reinstall /tmp/schema-b13-aka-dist-final/atrex_aka_schema-0.2.0-py3-none-any.whl
```

| 工件 | SHA-256 |
| --- | --- |
| `/tmp/schema-b13-aka-dist-final/atrex_aka_schema-0.2.0-py3-none-any.whl` | `5e4948e973242ef2b4e72388d63c2d71c1ee8c00735a11d2959bf47cfebafa0f` |
| `/tmp/schema-b13-aka-dist-final/atrex_aka_schema-0.2.0.tar.gz` | `e6ac31fe85701cd11a7ce2d46ce7ea3f6e17a03cc005a12c384b6507f3231c94` |
| `/tmp/schema-b13-aka-dist-final/index.json` | `1f4ff2519b3449c001e86228518a725846ee8cdd1c172b83db29daa6495d33db` |

wheel 的 **28 个 AKA 文件与源码逐字节一致**，原 LICENSE/NOTICE 字节相同，没有 KS 包或 private runtime。最终构建日志 `/tmp/schema-b13-aka-build-final.log`；最终安装日志 `/tmp/schema-b13-aka-install-final.log`。根的恢复增量之前已构建/安装的首版工件 `5f6522698c3ce409c1f085ff979ac15b1e249ac903e980ebe192ebfcdf73e6a1` 保留在 `/tmp/schema-b13-aka-dist`；首版 3 项 installed-child **8.14s** 记录亦保留，不作为最终工件证明。

```sh
/tmp/schema-b05-a3-venv/bin/python -I -B /tmp/schema-b13-installed-provenance.py wheel
/tmp/schema-b05-a3-venv/bin/python -I -B -m aka.workflow.schema.native_probe --help
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/schema-b05-a3-venv/bin/python -B -m pytest tests/test_schema_native_limits.py::test_probe_real_synthetic_child_retains_scope_locator_digest tests/test_schema_optional_module.py::test_fresh_child_installed_metadata_and_resource_source_pair tests/test_schema_optional_module.py::test_stdin_exec_preserves_exact_payload_in_real_child -q -x --tb=short --basetemp=/tmp/schema-b13-installed-child-final
```

最终 `/tmp/schema-b13-installed-provenance-final.json` 从 `/tmp` 的 isolated fresh process 证实 Framework/Legacy/Light/Schema 四包均从 venv `site-packages` 导入、字节分别与固定源码一致；KS module/resource 保持本 AKA 固定 1c9 子模块。四包均 0.2.0，KS0.1.0；实际 runtime 依赖 packaging26.2 / PyYAML6.0.3 / jsonschema4.26.0。`native_probe --help` 显示新参数且 exit0。

检查结束仅把 Schema 恢复原 editable 开发安装（`--no-index --no-deps --no-build-isolation -e .../packages/atrex-aka-schema`），其余三包仍原 wheel；日志 `/tmp/schema-b13-editable-restore-final.log`。恢复来源记录 `/tmp/schema-b13-editable-provenance-final.json`；没有恢复、执行或改动旧真实失败 scope。

## 原真实事实及未验证边界

- 本会话累计真实模型调用仍为 **7**；K3 历史 4 次的已知 tokens 下界为 **15291**、CLI USD 已知部分 **0.232672750000000010**。另旧 AKA coder、KS design planner、本轮先前 AKA coder 各 1 次 120 秒 timeout，tokens/USD 均 unknown，因此整体 tokens/USD 仍 **unknown**。GPU jobs **0**。
- 两次 AKA 运输 scope 均 coder1/reviewer0、120 秒失败且无 qualification。最新 e867 固定源码的实际输入 **355182 bytes**，原 owner active **124139 ms**、进程已终止、`qualification=null`；没有重试、continue 或新 owner 真实调用。
- 原 Schema wheel **`69962cda200cfd0240d686128642223f084eb2b21235d440eca5b76ce09e4d4f`** 与只读原版本保全仍保留；本修复不回写旧 scope、旧费用或旧结果，不声称它已获得新 ceiling。
- 本批未重跑完整 CPU suite，也不把历史 155 passed（KS76 配对）或最终旧版 26+5（KS1c9 配对）重新归为本次源码全库结果。
- 大于 120 秒的真实模型终态、native coder/reviewer 真实资格、完整 GPU Task/current-R fidelity/production、A6.real/A7.fixed/A7.pair、Agate auth/dev/readonly/same-allocation、Python3.11 仍未验证。保持原 AKA row-softmax/single_file/no-tools 支持范围。
