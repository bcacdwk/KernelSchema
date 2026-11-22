# B03 / K3：有限 Claude CLI 单调用、角色投递与终态恢复

工作树：`KernelSchema/.worktrees/schema-workflows-v1`，分支 `feat/schema-workflows-v1`；输入 core 固定为 `c1d7bd9cfe941e7440e6f3f8a2c6ae8d9b20bf52`。本次没有 Git 提交、推送或配置变更，没有修改 core contracts/api/artifacts/materials、共享 fixture、K1 evaluator 或业务循环。

## 交付与资格分层

| 层 | 结论 | 依据与限制 |
| --- | --- | --- |
| K3.interface | 已实现 | 版本化 profile；显式 requested/reported model；`admit/invoke/reopen/settle`；角色化完整输入；typed source/draft/review；原始事件/用量/失败收据 |
| K3.cpu | **46 passed，75.37 秒** | 真实 subprocess fake CLI；预算拒绝零启动、权限伪造、截断/非零/缺终态/缺用量、原操作恢复、source 续写、失败闭包导入，以及签收 HOME/公开 profile 收尾验证 |
| 可复用 CLI profile 的真实资格 | 三角色通过 | 同一最终 CLI/profile 配置的 coder、reviewer、planner 各一真实调用，均回收原操作并验证工具/cwd/角色材料边界；原先另一次 planner 被正确拒绝，失败与费用保留 |
| 新增写作输入语义 | **仅 CPU** | 真实资格结束后，按 supervisor 明确指令修正公共角色投递：implement 固定 R；design/optimize 可在 Task/硬判据内提案；reviewer 只审阅；workflow_objective 独立且按角色授权。未再调用模型。实际写作语义由后续 K5.real 验证 |
| Kernel / evaluator / fidelity / 完整业务迭代 | 未声明 | 真实材料是小型 synthetic transport Task，未生成、运行或优化实际 kernel，没有 GPU/Task qualification |

上述真实记录证明未变化的 CLI 权限、运输、终态与原操作定位能力；每次记录保留当时源码、profile、完整输入和实现摘要。它们不是“最后一次 CPU 语义修改已经实际写作成功”的证据。修改实现后直接继续历史私有 run 会按合同拒绝身份变化；保留的 `runtime-source/` 是对应原安装的源码归档。历史原始收据可用 core Bundle 独立读取。

[增量对照](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/qualification/20261120/cpu-input-increment.json) 核对了源码/AST：仅 `prepare_role_input` 与 `ClaudeRuntime.admit` 改变；system prompt、命令构造、native parser、invoke/settle/reopen、supervisor 与本地 PID 模块均未改变。

签收收尾另修正未实测的 `environment_auth`：`ClaudeRuntime._environment` 不再把 HOME 改指 config，而是原样传递调用者 HOME，仅用官方 `CLAUDE_CONFIG_DIR` 和 bare/restricted/空设置源隔离配置。`existing_cli_auth` 的分支正文与行为保持不变。未新增模型调用，历史 capture/source/原增量记录均不改写；相对真实 capture 的当前三个变化函数在可提交公开 summary 中明确列出。

## 文件与稳定 API

- [运行时与角色准备](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/claude.py)
- [导出 API](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/__init__.py)
- [有限 detached supervisor](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/_runner.py) 与 [本地持久/PID 工具](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/_local.py)
- [接口、输出协议、权限与恢复边界](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/INTERFACE.md)
- [可提交的公共 profile 与说明](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/profiles/README.md)：K4/K5 无需私有 capture 即可加载原已验证 CLI 配置
- [CPU 测试](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/tests/workflows/test_claude_runtime.py) 与 [专用 fake CLI](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/tests/workflows/fixtures/runtime/fake_claude.py)
- [显式单次资格工具](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/qualify.py) 与 [四次真实资格索引](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/runtime/qualification/20261120/index.json)

K4/K5 使用 `ClaudeProfile`、`ClaudeRuntime`、`CallHandle`、`CallReceipt`、`RoleDelivery`、`RoleOutput`。没有新增业务 backend，也没有复制父预算或多 agent 调度器。

1. 构造固定 profile；`profile.backend_profile(environment_ref)` 可生成规范 backend profile。包含它的业务 profile 必须固定 `parameters.claude = profile.as_dict()`，`model` 与 requested model 相同。业务 backend 自行收窄实际 operation 能力。
2. owner 先持久预约。`runtime.admit(context, operation_id=..., role=..., purpose=..., reservation=..., source=..., source_roles=..., deliveries=...)` 冻结输入与 core/native intent。`reservation` 是 owner 的原授权事实，不能由 runtime 创建新余额。建议明确带本调用获预约的 `active_ms`。
3. `runtime.invoke(context, handle, authorize=owner_launch_check, wait=True)` 在启动点调用 owner 检查 pause/fence/deadline/原授权；拒绝时零进程。实际 timeout 取 profile、固定请求与预约 active_ms 的最小值。
4. `runtime.reopen(context, operation_id)` 找回原 handle；`runtime.settle(context, handle)` 只吸收原事实，不启动 CLI/认证查询。相同 handle 再 invoke 也只回收原操作，不重复收费。
5. 终态 `receipt.usage_row()` 是规范 usage-index 行。owner 按 producer/operation/receipt_id 吸收一次；在途查询没有不可变终态收据，调用该方法会拒绝，owner 保留原预约/unknown 记录。`receipt.output` 只是待业务验证的模型数据。

输出协议是 `ks.claude.output.v1`：外层只允许 `protocol/role/assignment_sha256/status/kind/content/reason`。coder 的 `candidate_source` 是完整 UTF-8 源码字符串，`RoleOutput.candidate_source` 给原字节；planner 的 `writing_draft` 和 reviewer 的 `review` 是 JSON 对象。运行时不把其中的路径当成宿主写入位置，不把内容中的 pass/model/permission/usage 当成受信事实。`status=blocked` 与运输成功分开。

K4/K5 的 fidelity/writing 方法身份可使用 `facts.requested_model`、唯一的 `facts.reported_models` 元素（多值/未知需如实处理）、`scope_id` 与固定 runtime implementation identity。本次实际 reported_models 均只有 `qwen3.8-max[1m]`。

## 权限与实际 CLI profile

只读检查确认 `/usr/local/bin/claude`，CLI 自报 **2.1.316 (Claude Code)**。requested/reported 模型均明确记录为 **qwen3.8-max[1m]**；这不是 Anthropic 模型资格声明。最终 profile 摘要是：

`sha256:76fd98e0094f4a3e07b849c8f7a9207752689dd236105bd82919d904b405603e`

CLI 实际原命令保存在每个 capture 的 `run/.runtime/claude/<operation>/launch.json`，并进入不可变原始收据。关键公开参数为：`--print --verbose --output-format stream-json --model ... --effort xhigh --session-id <固定UUID> --no-session-persistence --safe-mode --tools '' --permission-mode dontAsk --permission-prompts none --strict-mcp-config --mcp-config '{"mcpServers":{}}' --setting-sources user --settings '{"disableAllHooks":true}' --disable-slash-commands --no-chrome --prompt-suggestions false --max-budget-usd 0.50 --system-prompt <固定正文>`。

这是 **host-delivered/no-tools** profile：源码/资料/反馈以完整文本从 stdin 投递；不授予模型文件读取、编辑、shell、网络工具、MCP、subagent 或任意宿主路径写入能力。实际 CLI init 必须回报 tools=[]、MCP=[]、skills=[]、dontAsk、准确 role cwd；任何 tool-use 事件、用户插件或错误权限/模型/会话/cwd 均拒绝输出。权限控制来自 CLI 实际工具集配置，不只是一段 prompt。

`safe-mode` 仍保留 CLI 内建插件。最终仅白名单本次实际看到的 `cc-plugin-agents-md`、`cc-plugin-plugin-authoring`，且 metadata 的 path 必须为 `builtin`；不泛化允许用户插件或 skills。role cwd 是 `/tmp` 下的空私有目录，不在仓库或 home 内；准入前检查自身为空、非 symlink，并拒绝存在 AGENTS.md、CLAUDE.md、.claude、.mcp.json 的祖先。实际 native cwd 与该 locator 核对。

官方 `auth status` 证明既有认证在 safe-mode + user-only 设置源下可用；清空设置源/restricted/bare 时该现有认证不可用。因此真实 profile 选择 `existing_cli_auth`，由官方 CLI 自然使用既有认证，不手工提取/复制认证值，不修改全局设置。**它不宣称独立 config 根。** 另有 CPU 验证的 `environment_auth` 模式，保留调用者 HOME，仅用新的 `CLAUDE_CONFIG_DIR`、bare/restricted/空设置源隔离配置；这次没有真实环境认证资格。

所有实际负例文件均为专门生成的 dummy canary；没有使用用户真实隐私作探针。三次最终 profile 调用均证明未投递其他角色 dummy 内容、未将外部 dummy 字节带入输入、外部文件未变、禁止写入目标不存在、外部 role cwd 结束后仍为空。

## 终态、费用与持久恢复

收费前保存 run/operation/role/purpose/C/预约、固定输入摘要、session UUID、profile、原生目录/日志/exit/PID locator。再保存 launch fence。detached supervisor 在真实 subprocess 中运行；child 在 exec Claude 之前 fsync 自己的 PID、Linux start ticks 与 boot ID，stdout/stderr 直接归原 operation。

调用方死亡后，原 supervisor 继续保存终态。仅见 native result 但进程仍活着时仍在途；supervisor 死但 child 仍活着时新操作被 fence；PID 不明或只有 launch gap 时不得重提。只有实际 exit，或两条已知原进程身份均证明已消失，才记录本地终止。后者仍报 `exit_unobserved`，不会接受成功输出。timeout 实际 kill 原进程组并 wait，再记录终止。

非零退出、原生失败、截断/非法事件、缺 terminal、缺 usage/cost、错模型/会话/权限与伪造输出都保存失败原件。已知 usage 不丢，未知 tokens/cost 保留 null。已终止且统计 null 不自动变成在途；owner 对无法核定的受限余额继续 fence。调用计数、native num_turns、physical retries 分列；无 retry 字段时为 null，不伪装零。只有 launch 但尚无 child 事实时 calls=null、已知下界零，保留预约。

`token_enforcement=unsupported`，有限 token budget 在启动前拒绝。单调用上限 120 秒；native `--max-budget-usd 0.50` 已真实传入，但它是 CLI 的费用阈值，不宣称宿主美元硬上限。以下 USD 全部是 CLI 回报值，不是独立核验账单，也不代表供应商实际结算价格。宿主请求 active clock 由 owner 唯一维护，不能把本调用 active_ms 与父时钟重复相加。

### 没有 Schema/C/成功审阅时的失败闭包

失败不是靠伪造 report 才能保留。规范 `usage_index.entries[].receipt` 指向一个自包含 native envelope：stdout、stderr、角色输入、profile、真实命令、PID/exit 等**原字节**以 base64 和 SHA-256 放在 envelope 内；没有要 core 猜测遍历的隐藏外部 Ref。CPU 测试将一个缺 usage 的失败 design 调用只通过 usage_index 导入新目录，并逐项还原/核对全部原件。

同样另外保存逐个原文件 Ref，`receipt.evidence_refs()` 可供 K4/K5 成功报告显式放入 evidence/raw_receipts。K2 对未知 native JSON 保持 opaque；上层无需 core 改动，也不能假设仅因文件内长得像 Ref 就会自动展开。完整业务 result 仍由 K2 按原请求/主体闭包发布；这里的自包含收据解决无输出失败的调用事实承载，不把 runtime transport 说成 Task qualification。

## 实际调用账目与有界调整

最初上限为三次、每角色至多一次。首次 planner 发现 safe-mode 的内建插件与原 profile 声明不符，正确拒绝。supervisor 随后明确将总上限增为 **四次**，只用于在最终 profile 补一次 planner；三次最终 profile 均未复用未知旧操作。每次仍使用 timeout≤120 秒、native USD threshold 0.50；没有额外模型调用。

| 次序 / capture | 角色 / profile | 原终态处理 | tokens | CLI reported USD | 本地 active_ms |
| --- | --- | --- | ---: | ---: | ---: |
| 1 / planner | planner / 初始 e6f50a… | host failed：未声明的 builtin plugins；native usage 保留 | 4,606 | 0.07659250000000001 | 57,734 |
| 2 / coder | coder / 最终 76fd98… | succeeded；精确占位 C 运输，无 kernel 生成 | 3,257 | 0.045436250000000004 | 27,597 |
| 3 / reviewer | reviewer / 最终 76fd98… | succeeded；typed review 运输 | 3,914 | 0.059609499999999996 | 43,143 |
| 4 / planner-final-profile | planner / 最终 76fd98… | succeeded；typed writing draft 运输 | 3,514 | 0.0510345 | 34,222 |

总 host model calls=**4**，tokens=**15,291**，CLI reported USD 精确求和为 **0.232672750000000010**。各次 native_rounds=1，physical_retries 均为 null。所有终态成功收据均做了原 run/operation 的 reopen/settle，且当时不存在首份业务 result.json；无重复调用/费用。初始失败也做了相同原收据重读。

第四次正式调用前有一次资格 wrapper 参数拼写错误，被 argparse 在创建 capture/CLI 启动前以 exit 2 拒绝；新模型调用数为零。只读 help/auth 查询也不是模型调用。原始失败、不成功的第一次 profile 和这些边界事实没有删除或改写。

## CPU 验证与后续 owner 边界

验证解释器为现有隔离 core 环境 `/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python`，**Python 3.13.12**；没有下载/切换解释器，没有新增依赖，Python 3.11 验证仍按上层安排后置。

```sh
python -B -m pytest tests/workflows/test_claude_runtime.py -q
```

角色输入语义增量完成时，完整回归为 **43 passed in 74.08s**；对应历史记录不改写。签收 HOME/公开 profile 收尾后的最终完整回归为 **46 passed in 75.37s**。新增测试证明 environment_auth 保留调用者 HOME/全局环境、仅使用官方 config 根，existing_cli_auth 行为不变，以及从其他 cwd 加载公共 profile 时禁止读取私有 capture 或查询 CLI/auth。另从 `/tmp` 以禁止 `aka/torch/triton/cuda` 导入的独立 Python 进程读取四个真实 capture，核对 **36** 份 self-contained original byte sequences 的 SHA-256；新模型调用 **0**，没有新生成或测量。这是历史读取与独立安装边界验证，不重签历史 Task 资格。

真实 subprocess fake 测试覆盖：三角色与 typed 输出；K2 allowlist、private evaluator 拒绝、外部 dummy/祖先配置拒绝；原始输出/stdio/usage/PID；非零/截断/缺 terminal/缺 usage/缺 cost/错模型/错会话/错 cwd/错权限/用户与未知内建插件/工具事件；原终态重放；首 result 缺失和真正杀死调用方后的恢复；supervisor 死亡但 child 仍活着的 fence；timeout 后确证终止；错误私有 output/usage 或伪造“已结算”文件不能覆盖原收据；发布后 receipt cache 丢失重建同一收据；稳定 C 新 session 续写；pause/预算拒绝零启动；硬 token budget 拒绝；失败 usage-index 独立导入。追加公共角色语义测试还覆盖 design/optimize 的 proposal 权限、reviewer 只读职责、workflow_objective 按角色授予、implement 不受自由优化参考覆盖，以及 verify_only 拒绝 coder。

K4/K5 仍负责真实 owner 账本/累计与有限后续行动、领域 Schema/source 验证、Task evaluator、coverage/fidelity/writing_review、晋升与业务 result。当前接口没有整合所必需的 core shape 缺项。真实写作增量等待 K5.real，不能拿这里的 transport 通过代签。

## 可提交集合与私有归档边界

主代理应按下面的**精确路径**提交，不递归加入 `kernelschema/runtime/`：

```text
kernelschema/runtime/__init__.py
kernelschema/runtime/claude.py
kernelschema/runtime/_local.py
kernelschema/runtime/_runner.py
kernelschema/runtime/qualify.py
kernelschema/runtime/INTERFACE.md
kernelschema/runtime/profiles/README.md
kernelschema/runtime/profiles/claude-code-2.1.316-qwen3.8-max-no-tools.v1.json
kernelschema/runtime/profiles/qualification-summary.v1.json
tests/workflows/test_claude_runtime.py
tests/workflows/fixtures/runtime/fake_claude.py
```

本报告位于原 main 的授权 implementation 路径，是否随计划材料提交由主代理处理。local exclude/包发现排除也由主代理处理，本子任务不修改 Git 配置、排除规则或 pyproject。

`kernelschema/runtime/profiles/` 只有三份小文件，共 **9,343 字节**。profile 快照与原 final profile **同字节、同摘要**；summary 只含必要非秘密配置关联、四次状态/model/token/CLI-reported cost、原件摘要和适用范围。没有复制 auth status、原始流、PID/会话/owner 定位或 `.runtime`。这是一份**公开说明**，不是可导入的完整原证据包；摘要不能替代原件。加载该 profile 不需要认证查询、CLI 调用或任何未提交 capture。

`kernelschema/runtime/qualification/20261120/` 的完整 452 文件私有归档保持原位置、原字节（约 3 MB），包含完整原 run、身份、状态、源码及收据。它不属于上述提交集合，也不是 public profile 的资源依赖。完整 run 不移动；需要本地恢复时保留对应原安装/私有状态，公开说明不代替其恢复条件。签收前后用全部相对路径及文件 SHA-256 的聚合清单核对，基线清单摘要为 `803fcefb8169d062977ac3fa09c395d661854a8e574f20eebf85d05c649cba76`。

## 操作边界审计

首条文件名发现命令误将 home 一并纳入 `rg --files` 范围，曾返回被禁止仓库的几个 AGENTS.md 文件名；没有打开这些文件、读取该仓库文件内容、运行或修改该仓库。此后检索均限制到 KS、明确的本机 CLI 非秘密事实与自建 dummy 目录。这一处枚举失误如实保留，不能声称全程完全零目录访问。没有派生代理，也没有修改原提取、计划、AGENTS 或全局 Claude 配置。
