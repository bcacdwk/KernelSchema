# B43：共用的有限 Schema 版本循环

2026-11-21 UTC。已补上 KS 任意显式 Task 与 AKA switch 的轻量自动串联入口：task → design（或已有 Schema）→ 原 AKA implement → 真实反馈 optimize → 必要的新版本 implement。**本次完成代码、文档和 CPU 接线验证；没有运行真实模型/GPU闭环，不声明新候选正确或性能改善。**

起点为 KS `b6a012bdde8d8dfda0baa1fa87d8ec09178a96a9`、AKA `ef8dadadd9867f298826d52ea159cf70bef93060`。全部改动未 stage/commit/push，由主代理精确固定。模型调用、GPU jobs、安装、提交和推送均为 0；B41 冻结快照及运行记录未触碰。

## 最小接线

唯一新循环位于 KS `kernelschema/iterations.py`，由 `api.iterate(...)` 和 `kernelschema iterate` 暴露。它直接调用现有 file `api.design/optimize` 和 `api.implement`；后者继续 lazy 委托 AKA `standard.implement`。AKA `schema/iterations.py` 只有 CLI 参数适配和一次 `api.iterate` 调用，没有复制循环或新增反向 submodule。

AKA 新分支必须同时显式使用 `--schema --schema-iterate`；必需 `--schema-task FILE` 与 `--schema-out NEW_DIR`，可以指定 `--schema-runtime-profile FILE`、`--schema-rounds N`、有限参考/目标/环境文件。添加 `--schema-path` 则从已有 Schema 开始，否则先 design。固定 `--schema --schema-path` 保持原多轮 coder 行为；默认关闭和 `--no-schema` 不导入 KS/optional Schema。旧 `--schema-config` 仍是分开的 legacy 入口。

`rounds` 是最多 Schema 版本/native campaigns 数，默认 2；`max_iters` 是原 AKA 在每个版本内的 coder 轮数，互不替代。默认两版本至多调用一次 optimize，`keep` 直接停止，不在每个 coding episode 后重写 Schema。design/optimize 仍是一 planner 写文件，没有独立 review、效果验收或新资格门。

AKA 普通 CLI 的 `provided_settings/provided_runtime` 决定 profile 默认值，不能先转成全部显式字段。新适配转发原 `WorkflowLaunch`，standard 只替换该版本的 workspace/name、启用既有 exact_workspace 并清除旧 restart arguments；原 ProblemSource、workspace_base、settings/runtime、provided-field 信息和 composition patches 保留。原 Light/Agent/工具/Executor/Evaluator/Agate/ABBA 都继续由现有路径提供。

没有默认自定义 Owner。显式 `owner_account/owner_limits/owner_run_id` 继续传入既有 native adapter；新循环不创建共享账户、预算/资格/调度/迁移或 continuation 框架。写作本身沿用已存在的有限单调用 runtime。

## 结果与停止行为

- 每个版本使用新的 `implement-NNN` workspace，写作使用独立 `design-NNN` / `optimize-NNN` 目录。
- `iterations.json` 列出各阶段、原结果路径、反馈来源/副本和停止原因；另存每阶段完整返回 JSON，原 native workspace/写作产物不删改。
- 所有声明的 native `feedback_files` 按字节复制，完整 native 返回 JSON 与这些副本一起交 optimize。已终态的实现失败照样送 writer；没有根据 exit 0、`completed`、旧 incumbent `PASS` 合成新 candidate 成功或 qualification。
- native blocked/pending/interrupted 或显式 unsettled usage 停止；缺失、越出原 workspace 或不可读的已声明反馈明确停止，不丢掉后继续。standard 对原 `ENVIRONMENT_TEMPFAIL` 返回值标 blocked，对 130/143 标 interrupted，以免环境临时失败被当普通终态失败送下个收费阶段；原 exit code 和 native 记录保留。
- 写作未 completed、未结算、没有产物或产物目录不可用时停止并保留诊断；`keep` 只在 completed 且已结算时结束。原 file writer 的基本文件/解析职责不变。
- 输出目录必须新建；已有空目录、非空目录、已完成或中断输出均拒绝，不自动重放可能已收费的阶段。异常/中断前的阶段入口和错误会留在列表；SIGKILL 等骤停仍会留下已创建的输出和 started/running 阶段，重调也无法自动重投。需要原操作结算时使用它自己的记录和既有操作入口。

## 公开用法

KS 的直接入口示例：

```sh
kernelschema iterate --task /materials/task.md --out /runs/new-iteration \
  --runtime-profile /materials/claude-profile.json \
  --problem /bench/problem --repo-root /checkouts/atrex-kernel-agent \
  --platform L20D --rounds 2 --max-iters 4
```

AKA 的入口示例：

```sh
aka optimize --repo-root /checkouts/atrex-kernel-agent --problem /bench/problem \
  --platform L20D --framework Triton --max-iters 4 \
  --framework-baseline never --convert-after 0 \
  --schema --schema-iterate --schema-task /materials/task.md \
  --schema-runtime-profile /materials/claude-profile.json \
  --schema-rounds 2 --schema-out /runs/new-schema-run
```

两者均可添加各自的 `--schema` / `--schema-path` 从已有设计起步。KS API 的 native settings/runtime/profile/patches 等参数继续传 standard；KS CLI 增加可选原生 profile/patch 和 settings/runtime JSON 文件接点。AKA 保留其原 CLI 参数。具体选项见更新的 standalone、WRITING 和 AKA standard-workflow 文档。

B42 vector-add README 已用同一入口替换手工串联片段，并给出 KS/AKA 命令。`task.md` 与 `root.yaml` 未改；root 仍是 9,106 B，SHA-256 `58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c`，没有宿主修稿或效果重验。命令是使用说明，不是收费闭环成功记录。

## 有限 CPU 检查

使用现有 `/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python`，`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，显式 `PYTHONPATH` 指向当前 KS 工作树和 AKA/src；没有安装、改包或访问运行中的收费环境。静态检查再次确认实际 import 路径是这两个当前源码目录。系统 `/usr/local/bin/python` 不含 pytest，最初两条 pytest 命令因此未运行测试，随后改用已有解释器。

1. 新 KS 循环最初 19 tests 全过；补充缺产物/未结算/已有目录边界后，运行新文件及原 `test_standard_implement_api_and_cli_forward_to_aka`、`test_file_cli_runs_without_request_bundle`、`test_extract_cli_still_calls_original_bridge`，最终 **27 passed in 5.01s**。日志与 JUnit：[ks-tests.log](/tmp/schema-b43-lightweight-schema-iterations-20261121/ks-tests.log)、[ks-tests.xml](/tmp/schema-b43-lightweight-schema-iterations-20261121/ks-tests.xml)。
2. AKA 新 `test_schema_iterations.py` 加既有 `test_schema_standard.py` / `test_schema_off.py`：**21 passed in 25.93s**。验证原 Light 两 episode、真实 composition boot、原 adapter failure/result、off 禁止 optional import/I/O，以及新入口参数透传、默认 off 连新写作 flags 也不导入 KS。此首次输出保存在本任务工具返回中。
3. 最后保留原 workspace_base 并更新错误文案后，只重跑相关 native launch 保留及显式参数错误两项：**2 passed in 1.14s**。日志与 JUnit：[aka-followup-tests.log](/tmp/schema-b43-lightweight-schema-iterations-20261121/aka-followup-tests.log)、[aka-followup-tests.xml](/tmp/schema-b43-lightweight-schema-iterations-20261121/aka-followup-tests.xml)。未为记录日志重跑整套。
4. 8 个改动 Python 文件 AST、17 个文档 shell/Python 代码块语法、两条新 help、精确路径 `git diff --check`、root 原字节 SHA 全过。[verification.json](/tmp/schema-b43-lightweight-schema-iterations-20261121/verification.json) 与 [check_sources.py](/tmp/schema-b43-lightweight-schema-iterations-20261121/check_sources.py) 记录本地检查。静态脚本首次因局部变量被 `import aka` 覆盖而中止，改名后通过，未影响产品源码或运行调用。

这些检查使用现有 API/native seam 的 mock 或既有 CPU fixture。覆盖从 task 先 design、已有 Schema 直接 implement、失败反馈原文交 optimize、keep 停止、版本数与原 max_iters、原文件保留、missing output/feedback、blocked/unsettled、中断与已有输出不重放、AKA off 无 KS 导入。没有效果/独立 review 验收，没有运行旧 strict 大套，也不推定 B44 环境兼容或未来实际候选已通过。

## 精确文件清单与交接

仅以下 12 个产品文件和本报告（原 main 指定 discussion）有本任务改动；不涉及原 main 算子提取、AKA execution/measurement 底层、submodule/release 指针、B41 快照或旧账。没有读取、枚举、运行或修改禁止实验仓及 archive 变体，没有 home 递归搜索。

| 仓库/位置 | 文件 | 用途 |
| --- | --- | --- |
| KS 工作树 | `kernelschema/iterations.py` | 唯一有限循环 |
| KS 工作树 | `kernelschema/api.py` | lazy `iterate` API |
| KS 工作树 | `kernelschema/cli.py` | iterate 命令及原生选项透传 |
| KS 工作树 | `tests/workflows/test_iterations.py` | CPU 接线与停止边界 |
| KS 工作树 | `docs/workflows/standalone.md` | 任意 task 有限入口及旧固定示例区分 |
| KS 工作树 | `kernelschema/WRITING.md` | 新入口链接与版本粒度 |
| KS 工作树 | `examples/workflows/vector_add/README.md` | B42 公共示例新命令 |
| AKA | `src/aka/bootstrap/workflow_cli.py` | 显式自动写作分支；off/固定分支保留 |
| AKA | `src/aka/workflow/schema/iterations.py` | 仅参数适配，调用 KS 共用循环 |
| AKA | `src/aka/workflow/schema/standard.py` | 保留原 launch；识别环境临时退出/中断 |
| AKA | `tests/test_schema_iterations.py` | CLI/原 launch/off CPU 检查 |
| AKA | `docs/schema-standard-workflow.md` | 新 switch 的可运行用法与边界 |
| 原 main discussion | `implementation/b43_lightweight_schema_iterations.md` | 本报告 |

[source-manifest.json](/tmp/schema-b43-lightweight-schema-iterations-20261121/source-manifest.json) 列出 12 个产品文件的精确路径/字节/SHA；[change-manifest.json](/tmp/schema-b43-lightweight-schema-iterations-20261121/change-manifest.json) 另外包含本报告。AKA 的新 `tests/test_schema_iterations.py` 命中既有 `.gitignore:19:tests/`，需主代理对**这一精确路径** `git add -f`；没有修改 ignore 文件。

由主代理按此清单固定源码，再统一处理 KS gitlink/release 配对与最终 delivery 文档。本任务不提交推送，也不扩展真实调用。未来运行需要使用匹配安装后的两仓版本、明确写作 runtime 和原 native 资源配置；已有 CPU 结果不替代新候选的真实 native 评测。
