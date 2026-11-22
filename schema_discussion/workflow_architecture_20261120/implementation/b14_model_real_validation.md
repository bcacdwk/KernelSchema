# B14：长 profile 下的真实模型与 AKA 验证

2026-11-21 UTC。本阶段先做零模型/GPU调用的准备，等待唯一 B14 parent、固定 KS 修订及 supervisor 释放串行收费 slot。**尚未获得本轮 native transport 或业务成功证据。** 历史三个 120 秒失败 scope 不重试、不修改、不以新账户抵销；过去 7 次实际外部模型调用和未知费用继续归属原记录。

生产源码唯一可写仓为 `/home/shaohanyong.shy/atrex-kernel-agent`，开工 HEAD `804e26f79d094250a5852ad2b8a6f652f03f2928`、KS detached gitlink `1c9e592c6e0ec021f91e17af5a8c39956ce39178`，工作树干净。本阶段未修改 AKA 代码；仅新增本报告和 `/tmp/schema-b14-model-real/` 私有验证资料。未访问、枚举、运行 AKA_experiment，未改 KS 源码，未提交、推送或派生代理；没有安装另一代理的活跃工作树。

## 已完成的零调用检查

- 已读适用 AGENTS、progress/next_steps、B10 AKA/K5、B11 和 B13 报告，以及公开 source-release/dynamic-workflow、KS runtime INTERFACE 和 WRITING 的输入/输出、deadline、身份与账户相关段落。
- 从 `/tmp` 使用既有 `/tmp/schema-b05-a3-venv/bin/python -I -B` 核对实际 import。Schema 来自本 AKA editable；其余三个 AKA 包来自同源 installed wheel；KS import/resource 仍为固定 1c9 子模块，`verify_release()` 通过。证据：`/tmp/schema-b14-model-real/initial-installed-provenance.json`。未重跑完整 155 项 CPU 测试，未安装或下载 Python。
- 官方 `ClaudeProfile.inspect` 只运行 `--version/--help`；实际 CLI 仍为 `2.1.316 (Claude Code)`，executable SHA256 `fd13c70000fec3f2116dfbcbb6121210b4a29328bf0e46210a59dd594a67d824`，与原公开 profile 完全一致。
- 官方 `claude auth status` exit0，`loggedIn=true`。只让官方 CLI 使用既有认证，不读取、复制或输出凭证，不改全局配置。认证原件只在本项私有 `/tmp`，不进入任何模型角色材料。
- 新 profile 唯一变化为 `timeout_ms:120000→600000`。文件 `/tmp/schema-b14-model-real/claude-code-2.1.316-qwen3.8-max-no-tools-600s.b14.json`，JCS SHA256 **`f86190b7c579afe95fede90ecc762670297293b283249db6d3d181f5533cb0f0`**。保留 `qwen3.8-max[1m]`、accepted reported names `qwen3.8-max[1m]/qwen3.8-max`、effort `xhigh`、existing_cli_auth、原 builtin 名单、USD0.50 threshold、2MB 输入/16MB 输出与 no-tools/MCP。该新身份已交同轮 GPU 代理供 `Demo.create` 复用，未冒用旧 120 秒资格。

实际命令：

```sh
/tmp/schema-b05-a3-venv/bin/python -I -B /tmp/schema-b13-installed-provenance.py editable > /tmp/schema-b14-model-real/initial-installed-provenance.json
/tmp/schema-b05-a3-venv/bin/python -I -B /tmp/schema-b14-model-real/inspect-profile.py
/tmp/schema-b05-a3-venv/bin/python -I -B /tmp/schema-b14-model-real/prepare-native-scope.py --help
```

`inspect-profile.py` 的摘要 `/tmp/schema-b14-model-real/profile-preflight.json` 仅含非秘密配置、logged-in 布尔值和零调用事实。

## 角色协议检查与新请求准备

当前 `native_probe.capture` 已有仅运输指令：coder 原样返回 current_source，reviewer 返回 `transport_probe=true` 和“Task and fidelity not tested”。但该指令仅作为 `delivery/native_transport_probe` 普通 text 材料交付；默认 `fixed_assignment.workflow_objective` 仍要求完整实现、所有硬判据与性能目标，而 system prompt 将材料中的指令视为数据。故旧 120 秒仅 init/thinking 不能直接证明模型无效；需承认目标表达可能使模型继续进行实际任务推理。

本轮准备脚本通过既有的 Materials `workflow_objective`→`fixed_assignment.workflow_objective` 授权投影，为**新的运输 Q**明确冻结仅运输目标。它保留原 Task、R、scope、Rules、seed、所有其他必要材料和声明 profile，不截断语义文档、不造模型响应。外层输出仍必须是精确的 `ks.claude.output.v1` 七字段 JSON；完整资料正文/alias 将用原 `RoleContext.validate`、`material_text`、逐项 SHA/size 复核。预览身份与实际运行 capture 身份将分别记录，不能把 preview 冒称实际投递。

准备入口 `/tmp/schema-b14-model-real/prepare-native-scope.py` 已完成 `--help` 导入检查，**尚未对 Task 运行**。其参数为 `--base-request <GPU代理冻结的fixed.request.json> --authority <唯一B14 owner.json> --out <新的本项prepared目录>`。它只导入该请求的明确闭包并发布新的请求，不创建 parent、core run、预约或费用操作；对 authority 仅只读核对 descriptor，不打开 OwnerAccount 计时。原 GPU 代理业务 Q 不修改。

运输 Q 本身不用作业务选择输入。成功后使用原包含真实 K6 资格的业务 fixed Q 调用 `native_probe.select_request`，把实际 capture 的 input/exec/invocation 作为 `roles=[]` host-only Materials；业务保持原目标和原 Task/R/parent/exact 600s Claude profile。KS writing 也使用这个相同 Claude profile 和同一 parent，但其实际 planner/reviewer 结果不能由 AKA capture 代签。

## 本轮建议的有界运行政策

- 唯一 parent 由同轮 GPU 代理创建，总上限 model_calls10、evaluator_jobs8、active3600000ms、tokens=null；角色上限不改变总上限。所有收费操作串行，等待 supervisor 明确释放 slot 后才运行模型。
- 新 native scope ceiling 固定600000ms；coder/reviewer 各至多1，总至多2，首失败即终止本 scope，不 retry。新的运输 Q/native 上限同时为 planner0/coder1/reviewer1/extract0/model_calls2/jobs0/active1250000ms/tokens=null。
- profile `minimum_active_ms` 计划固定 coder600000/reviewer600000/evaluator1000，coder admission保留完整600000ms reviewer最小容量。额度不足就拒绝；不换 parent、不压低收尾容量来掩盖余额不足。该最小值只保留实际准入所需容量，不是凭空费用。
- 实际 launch 仍取声明profile、scope、原 request、原 parent/native 剩余额度、原 UTC/monotonic 截止的紧限；宿主固定5秒仅收集终态，不续领。USD0.50是CLI原生阈值，非宿主美元硬保证；无硬token声明。
- 必须核对 actual no-tools init、allowed model、输出/assignment、完整usage、原 PID/boot/start-tick/locator 与exec绑定。若缺terminal/tokens/USD，保存unknown，不按0结算；未返回进程保留原fence并只settle原操作。continue不能覆盖scope或重试已付费调用。

## 当前真实账目与阻塞

本项新增 **MODEL0/GPU0**，没有 B14 模型收费操作，也没有第二parent。原会话历史：K3四次已知15291tokens、CLI USD0.232672750000000010；另旧 AKA coder、KS design planner、新 AKA coder各一次120秒超时，tokens/USD未知、reviewer未启动。总历史tokens/USD仍unknown，不能只报已知部分为总额。

当前待交接：GPU代理固定新KS SHA、唯一B14 parent和真实H20 Task/fixed Q，以及K6资格结果；随后根提交固定SHA，再更新AKA gitlink/release并完成真实安装来源核对。收费slot尚未释放。Agate官方认证已由根验证成功，旧401不再作为当前阻塞；remote GPU/dev/readonly/full判据资格归同轮GPU代理实际结果。完整design/implement/optimize/dynamic与GPU闭环仍须各自真实证据，准备JSON不算闭环。
