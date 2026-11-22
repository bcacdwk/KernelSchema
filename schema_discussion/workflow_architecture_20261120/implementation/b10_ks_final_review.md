# B10 / KS 独立最终工程审阅与修复

2026-11-20。审阅及源码工作树为 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`，分支 `feat/schema-workflows-v1`，审阅起点及本代理冻结源码时 HEAD 均为 `8e846d7dd6321dc226436ac8fc80fbf569b0965f`。开始时工作树干净。本报告是原 main 唯一写入；未重做计划正文、提交/推送/配置 Git，未派生代理，未访问 EXP 文件或运行 EXP，未修改 AKA。原 Schema/kernel/提取/六算子/FA4 原材料、AGENTS 与私有 CLI 归档未改。

已读 AGENTS、公开 protocol/standalone/target-environment/FA4 README，四份 core/runtime/implement/writing 接口，现行 C/V、progress 与 B07/B08/B09 KS 报告。本批完成的是独立 CPU 工程审查与明确缺陷修复，**不是 MODEL/GPU/FA4 原栈/fidelity/性能/发行许可签收**。新增真实模型调用、GPU/Agate 提交均为 **0**。历史 K3 四次 transport 调用与其旧失败、另一次 AKA coder 120 秒超时及后者未知 tokens/cost 保持原事实；未执行或改写旧 capture。

## 已复现并修复

| 级别 | 证据、修复与界限 |
| --- | --- |
| 高 / 资格账户恢复 | row-softmax 与 FA4 的 qualify 原来无条件 `for_context(create=True)`。独立临时 synthetic 反例中，已计原 1 job 后移走原 binding+authority，保留资格 state/records，重放返回 0 jobs 并新建同 ID 账户。保留 binding 但移除原 operations 也未被拒绝。现两入口共用 `channel._qualification_account`：先持久化一个私有 initialization intent；重入仅开原 binding；核每个原 core intent 的 reservation 与原 owner operation 的 JCS 字节一致；丢失资格 state/native 对应关系拒绝。最终另用已留存 native 终态逐一核回原 core intent、原 reservation 和已终、非空的原用量历史，堵住 operations 与 core intents 同时丢失的组合。初始化 intent→binding 之间中断也保守拒绝。它不是新 operation、余额或账本；未修改 OwnerAccount 格式/API。FA4 selection 同时固定其本来已依赖的 `channel.py` 字节。 |
| 高 / 重建资料边界 | 原检查只挡 baseline 整个 C 摘要。把原 `operator.py` 的 **28,587 bytes** 改名为 text，或包进不同 Source manifest，schema_rebuild coder 均能获得。现统一 `materials.rebuild_source_hashes` 对该 intent 的**非空 baseline Python 文件原字节**拒绝 Materials/Rules/text/Source 的改名、改 manifest 投递。每项仍先核权限/私有集合/UTF-8，再投影/去重。空 package marker 不泄漏实现，继续合法；original_verify/明确 seed_continuation 及原真实 row-softmax baseline Source 例外仍合法。这是完整文件字节身份检查，不是改写/截取代码的通用污点检测或“模型未见过原码”证明。 |
| 中 / fidelity 定位与坏输出 | `_located` 原来只需为声明位置找到一个正确 quote，其余 `missing.py:L999` 假证据可被忽略；unhashable id 会抛 TypeError，reason 对象会打断正式 report，额外 coverage 行也可被忽略。现逐项检查完整、唯一 location/evidence 对应、实际文件/行/quote；义务和 coverage 的字段/数量必须符合方法；非字符串 reason、坏 ID 等保留原收据并形成 unknown。没有以位置或自报 launcher 字段代签语义 fidelity。 |
| 中 / alias 一致性 | 手工构造同时带 `text_alias_of` 与冲突 text、不同 size 或指向 host metadata 的条目，旧 `material_text` 仍返回正文。现只接受同 payload 直接正文目标，检查原/alias size、SHA、UTF-8，拒绝冲突 text、host 目标与循环。生成投影仍保留每个原 ID/ref/category/authority/size/hash；未给 alias 继承目标权限。 |
| 中 / 当前请求资格投影 | 同 R 的新 verify 请求已固定、尚未返回时，driver.summary 原来按 R 相同显示旧 Q 的 qualified。现 pending implement 请求（含 verify_only）令 `current_qualified=null`，原 C/资格仍在 history；收到该请求结果后才投影其资格。D3、keep/revise/default_start/verify 的准入逻辑未改，也没有把普通失败变成设计阻碍。 |
| 中 / 观察导入的完整 stage | `freeze_observation` 原来只核列出的文件，额外 `source/extra.py` 未拒绝。现直接复用现有 `verify_stage` 完整集合/字节/mode 校验，删去重复的局部检查。观察导入仍一律 unknown，不因此成为合格 executor。 |
| 低 / blocked 空 usage 重发 | 缺 FA4 配置/资格的继续原本重发相同摘要、不同路径的 empty usage index。本批复现 result 不相等而费用仍为零。现仅在**本 run 无 operation**时持久化并复用原空索引 publication ID；每次仍检查原 owner、fence 与 channel admission，不缓存一个可过期的 passing 状态。没有重建账户、插入占位费用或制造资格。 |
| 文档 / 可复制源码使用 | FA4 README 改用明确 checkout 脚本路径，避免 `python -m examples...` 对 cwd 的依赖；真实 standalone 命令明确传入当地 inspected runtime profile，不能把旧安装的绝对 CLI 路径当可移植运行时。修正公开 FA4 admission 页末尾过时的“K3 未接入”结论；新增 `docs/workflows/release-gaps.md` 整理已有许可事实和精确未决项，不改原 K7 记录或授予新许可。 |

独立修前事实在 `/tmp/ks-b10-repro-4kipbv9e/summary.json`；独立 pytest 反例见 `/tmp/ks-b10-repro-pytest.log`、`/tmp/ks-b10-repro-additional.log`。这些是新建 synthetic 目录；账户丢失试验保留了其自身 `.retained` 原副本，未接触真实费用。

## 核过且保持的接口

- `contracts.py`、`artifacts.py`、`budget.py`、`writing.py`、原 Template/parser 未改；仍只有一个 codec、Source 实现及 OwnerAccount。完整 python_tree 的 32 文件/4 MiB/UTF-8/0644/有限路径、安装/删除/交换后恢复、旧 single_file 行为保持；typed candidate_files 和整树角色输入/实际上传身份由原入口验证。
- 私有 evaluator 集合仍由唯一 `private_evaluator_hashes` 定义 files+entrypoint+inputs，普通 Materials/text、等字节改名和假 baseline 不能绕过；timing/environment 可明确公开投递。未知 `.py` 规则没有按后缀删正文，明确获授的普通工具代码仍可读；Rules/tool 原闭包与身份保持。
- `writing_finish_floor` 仍是完整 cost 向量，拒未知/缺键/bool/不一致 model_calls；planner 加 reviewer 最低量，reviewer 保留 finish 向量，request/parent/native 取紧。默认独立 writing 的零 floor 保持，不新增 holding operation 或复制余额。A6 不需要变更该参数接口。
- FA4 仍只承诺 pinned 四节点 Schema 的 F1 B1/Q4/L64/P1/N1/S1，并保留 REST。原 13 ABI、BF16 randn×1.22/0.96→FP8 顺序、seed20261120、413138944 scratch、FP32 `0.06+0.04*abs(reference)`、原 out/x 身份/字节、冷启动及 5 warmup+40 full-forward 测量均未改。传输仍核实际完整 Source/evaluator/worker/command/key/profile；self-reported launcher 元数据不能替代 native 路径和 CuTe/TMA/TMEM/parity 的独立 fidelity。
- 独立 driver/design/optimize 仍是 row-softmax；reference implement 支持 row-softmax 与固定 F1。未知 variant/新 FA4 设计、通用源码树/恶意 native 沙箱未被扩大。缺 owner/bootstrap binding、原身份/费用不确定、过期预约/未知原 job 不能得到新预算或替代提交。

## CPU 验证与准确命令

使用已有 `k1-evaluation` Python **3.13.12**，固定 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。从 `/tmp` 先实测 `kernelschema.__file__` 指向本工作树；未使用旧 optional KS 污染环境，未下载解释器或大栈。

| 轮次 | 实际结果 |
| --- | --- |
| 唯一完整、稳定生产源码的原有 CPU 回归 | **380 passed，0 skipped，1504.60s**；发生在本批生产修复之前。新增审查测试在该轮 collection 之后写入，不冒称属于这 380 项。日志 `/tmp/ks-b10-full-cpu.log`。 |
| 初轮独立新增反例 | **14 failed / 2 passed，90.68s**；另补 ledger operations/初始化反例 **3 failed / 16 deselected，42.77s**。保留失败原件。 |
| 修复后第一批 19 个反例 | 初跑 **18 passed / 1 failed，128.82s**；失败是新 parametrized 测试共享 fixture 下复用输出文件名，尚未进入产品断言。仅加上 loss 类型区分测试路径后，原失败项 **1 passed / 18 deselected，23.66s**。生产代码未因该测试错误而改动。 |
| 修后受影响兼容组 | **56 passed / 128 deselected，468.56s**。日志 `/tmp/ks-b10-affected-cpu.log`。 |
| 额外正向 FA4 stage/观察/verify/seed 恢复 | **1 passed / 8 deselected，11.02s**；**3 passed，110.24s**。日志 `/tmp/ks-b10-fa4-observation.log`、`/tmp/ks-b10-fa4-handoff.log`。 |
| 最后同类组合丢失补查 | 修前 **2 failed / 19 deselected，31.54s**；补强同一 guard 后，两个新反例及 row/FA4 两个原资格正向 **4 passed / 45 deselected，78.67s**。日志 `/tmp/ks-b10-combined-loss-before.log`、`/tmp/ks-b10-combined-loss-after.log`。至此新增回归共21项，不重复全套。 |
| 静态与选定原文核对 | 15 个改动/新增文件的 Python AST、Markdown 本地链接及 `git diff --check` 通过；旧真实输入/投递副本原字节不变。 |

各组有重叠，不将数字相加冒充不同测试数；修后没有再次跑全库。最后 guard 补强后还重新生成了最终新目录安装/FA4 工厂 smoke，以下所列最终目录含最终源码字节；早期 smoke 目录保留，没有覆盖原件。

```sh
cd /home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest tests/workflows -q --disable-warnings --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest tests/workflows/test_review_regressions.py -q --disable-warnings --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest tests/workflows/test_review_regressions.py -q -k 'operations and fa4' --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest -q \
  tests/workflows/test_role_materials.py tests/workflows/test_fidelity.py \
  tests/workflows/test_fa4_bridge.py tests/workflows/test_fa4_channel.py \
  tests/workflows/test_agate_channel.py tests/workflows/test_standalone.py \
  tests/workflows/test_writing.py tests/workflows/test_claude_runtime.py \
  -k 'role or baseline or material or private or paired or same_C or fidelity or tree or rebuild or complete_measurement or six_job or qualification_same or qualification_unknown or qualification_entry or missing_qualification or incomplete_probe or concurrent_qualification or unknown_coder or driver_two_rounds or same_r or keep or finish_floor or same_parent or design_actual_api or default_factory or first_ready or two_rounds' --disable-warnings --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest tests/workflows/test_fa4_case.py -q -k 'observation or stage' --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest -q \
  tests/workflows/test_fa4_case.py::test_synthetic_success_never_promotes_qualification \
  tests/workflows/test_fa4_bridge.py::test_api_original_verify_current_whole_source_and_report_closure \
  tests/workflows/test_fa4_bridge.py::test_explicit_seed_and_no_first_result_recovery --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest \
  tests/workflows/test_review_regressions.py tests/workflows/test_agate_channel.py tests/workflows/test_fa4_channel.py \
  -q -k 'operations_and_intents or qualification_same_k1 or six_job_qualification' --tb=short
```

## 源码交付、真实工厂 smoke 与原输入

`/tmp/ks-b10-source-smoke-tsdqg7p4/` 保存 `source-snapshot.json`、`commands.json`、`summary.json`。明确挑出 **128 个**当前公开源码/资源文件，复制到全新 checkout 子集；没有 .git、private planning、capture、作者缓存或 AKA。使用既有解释器创建临时 venv，只以路径复用现有 CPU 依赖，再以 `pip --isolated --python <new-python> install --no-index --no-deps --no-build-isolation -e <copy>` 安装。未改已有 k1/k2 env。实际从 `/tmp` 导入并断言模块/规则资源来自新副本，import guard 拒绝 AKA/torch/triton/cuda/cutlass。随后五请求 prepare、fixed run、continue、export 均成功；blocked/new_request、calls/jobs/tokens/active_ms 全零，result 完全幂等，导出无 owner authority。这验证 row-softmax 支持子集；不冒充完整仓库资源包、FA4 无 Git 初始材料包或 wheel 资格。

`/tmp/ks-b10-fa4-smoke-u392ga40/` 保存最终真正 FA4 prepare + reference/FA4Channel 工厂的命令与 result，未注入 fake channel。请求具有 reviewer=1/evaluator_jobs=1/active_ms=60000 的非零预算、qualification=null，实际从 `/tmp` 运行公开脚本和 `python -m kernelschema implement`。结果 blocked/new_request，明确缺 probe/baseline/四坏候选原资格；C 仍为 `sha256:c8395325f884d636f9c2e0dd3b622680848e270c40cb55a8b9fb178190116611` 的 12 文件原 Source，task_uncovered=REST，calls/jobs/tokens/active_ms=0、unsettled=[]。继续后的**完整 result 与 usage Ref 相同**。早期 `/tmp/ks-b10-fa4-smoke-csneikno/` 的 smoke 检查脚本误将 CLI 人类摘要解析成 JSON，而真实 CLI 已成功发布 blocked；修脚本从原 result.json 读取，继续同一 run，未重开业务/模型/作业。

可复制路径已在公开 README/standalone 中给出：

```sh
KS=/absolute/path/to/KernelSchema
python -m pip install --no-deps --no-build-isolation -e "$KS"
python "$KS/examples/workflows/row_softmax/run_demo.py" prepare --out /tmp/softmax-demo --start fixed
python "$KS/examples/workflows/row_softmax/run_demo.py" run --demo /tmp/softmax-demo
python "$KS/examples/workflows/row_softmax/run_demo.py" continue --demo /tmp/softmax-demo
python -B "$KS/examples/workflows/fa4_decode/prepare.py" --out /tmp/fa4-f1-prepared
```

FA4 第一次准备须有两个固定 Git commits；代码没有隐式 fetch。已有导入的 Task/Schema/Source 闭包可再次 stage；新目录源码安装不等于已完成匿名固定提交取得或许可签收。实际 MODEL/GPU 命令还须显式本地 Claude profile、目标资料、原资格与同 owner 预算。

仅按授权读取旧请求的选定原文及 B09 副本，未执行 capture。`/tmp/ks-b10-role-check.json` 独立核 67 个原 ID 的 ref/category/authority/size/hash，38 个非 host 正文逐字一致；新 payload 69 项中的 alias 全部用当前严格 `material_text` 解析、重新核 bytes/size/SHA。输入仍为 **1,012,929 → 354,400 bytes（约减少65%）**，29 host 项省正文、13 aliases；未再生成真实调用。原输入 SHA `d03d9235e0e96fd1e50021578fe543f11fe5ce1bbcfdefc470d524310123183e` 与 B09 投递 SHA `989818c494f6b5f541e5e41c5cc898ff6081f3f9ec6cd3d9c977127466d62649` 均保持，不能把省字节解释为模型成功、token 结算或费用下降。

## 下游 A6/最终 AKA 与未签收项

1. **writing floor 接口稳定**，可直接使用现有完整向量；outer driver 的 writing request/parent/native 限额本身也须包含该容量，下一请求仍由同一 owner 明确准入。本批未改 writing.py 或 budget.py。
2. `material_text` 现在拒绝不完整/矛盾的手造 aliases。A6 应消费 K3 原完整条目的 size/hash/ref/authority，不要只复制 text_alias_of 或把 host metadata 当空语义规则。原 single_file `current_source.text` 仍保留；typed tree 接口没有变化。
3. schema_rebuild 的精确 baseline 文件禁入现在跨所有角色输入路径；正常明确 Source 授权与合法 row baseline 保持。它不是未见原码证明，也不扫描私有 histories。
4. qualification 初次绑定有原初始化 intent；已持久的 binding、account operations、core intent、已终原用量历史与 native records 必须保持可核对应。缺必要关联不能调用 create=True 重建；本批没有账户迁移。FA4 selection 必须由当前函数新冻结，新增其实际依赖 channel.py 的 hash。
5. 新 Q（包括同 R verify）pending 时 current_qualified=null；旧资格仍读 history。old/new R、迟到原 receipt、unknown fence、D3 与后续行动归属不能改成 native best/promotion 或复制旧 pass。
6. 所有修改会改变原实现摘要和相关 Rules/profile。旧真实 capture/运行必须保留其**原固定安装**做原操作恢复；不能热改旧 request/profile/authority 适配新源码。最终 KS commit 和 AKA pin 由主代理固定，本代理没有提交。

未修/未签范围明确保留：主 API anchor→原 owner binding 的 bootstrap 缺口仍保守拒绝，不提供新余额恢复；本有限 driver 无独立认证设计阻碍输入适配器；FA4 design/optimize、未知变体和 AKA 自身完整 F1 接线没有被本批实现。实际 model、GPU、原 CuTe/SM103/driver/compiler/TVM-FFI、readonly/完整 stdout/cleanup、真实 full-forward 与独立 fidelity 仍缺资格；CPU、历史回放、目标声明或自报 native 字段均不能代签。Python3.11、完整 wheel、匿名 FA4 两固定提交取得、项目 license/完整第三方 notices 仍是精确发行缺项。新 release-gaps 只整理已有 MIT、NVIDIA BSD/相邻 Apache/缺失 notices 事实，未擅自给存量或第三方材料授许可。

私有 452 份 CLI 档案未递归读取、复制、打包或提交；`git ls-files kernelschema/runtime/qualification` 为空，原 pyproject 的排除规则保持。所有源码改动已停止，待主代理固定。最终改动文件摘要如下（不含本报告，见 `/tmp/ks-b10-source-hashes.json`）：

| 文件 | SHA-256 |
| --- | --- |
| `docs/workflows/cases/fa4_decode-admission.md` | `a449a5c59595d57c0804052ca5aa68165cb7ffd2846d82c05fb5d92c70c6cfdb` |
| `docs/workflows/protocol.md` | `a8623bde8abecfac25fa05e10a50cdebca08a02ede2be80092182f43807f6f5c` |
| `docs/workflows/release-gaps.md` | `1fe954d1a8efd656017b929c9e6645e9cf890d9e910176790a79e1b381061205` |
| `docs/workflows/standalone.md` | `46f6e5452d715488aca7af161e402ea9319d7f381530bc452b113e40801b7ffb` |
| `examples/workflows/fa4_decode/README.md` | `0b1e6830f6587a1e8a9fa1daba56d5bb4972fb079032039a935cb452ae88719a` |
| `examples/workflows/row_softmax/driver.py` | `b6c366d610aebcff7409e77966eedb13513be7c1fb706efeb8283a9b1b1b1aea` |
| `kernelschema/evaluation/channel.py` | `f3dde3823f1f3b94d8fdc075a9127c172017f94bd4d2fc96381028a6dffb534b` |
| `kernelschema/evaluation/fa4.py` | `24d5c5b3499f028d0f6cdbdc25d6620f13c9b4244ccaffb03153c727bfb30584` |
| `kernelschema/evaluation/fa4_channel.py` | `6a6e761c7a8ff0f4adcb81ad5a9ca94fca2be98ea08bf69a9ef42b08c6ac026e` |
| `kernelschema/fidelity.py` | `624c99c0f69fa89826bd540031ef6eae43693f1fa26dc9b21feb7047f7458786` |
| `kernelschema/implement/reference.py` | `5769dbe43813bf49125fadc2fced8abc39cdf2940bb63462cb6d0f1c032b6915` |
| `kernelschema/materials.py` | `b177d25f391e4263443c9953d9887cc58b6f96ac9861eaeae7ff0c93388990ff` |
| `kernelschema/runtime/INTERFACE.md` | `357a1508d96a145a409d06f2d98d452ac0327770a47f692bd6704a7c8aacf76c` |
| `kernelschema/runtime/claude.py` | `3a0ceca91e7ed340fe7f356cf2ec8d9d3c93119556c2ca03e42f537144a4dad6` |
| `tests/workflows/test_review_regressions.py` | `a544bee8627eded8755b1924a611c18a5c4981918ee968c21f39e61d0a14a3e8` |
