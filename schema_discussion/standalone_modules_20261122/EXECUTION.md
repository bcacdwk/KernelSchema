# 实施与真实运行记录

实施依据：[FINAL_PLAN.md](FINAL_PLAN.md)。用户已授权计划定稿后实施、限定清理、commit，并持续排查到三个独立功能和 Workflow 真实跑通。

## 当前状态

- 计划：max 调研与具体计划、high 精简、xhigh 终审已完成，过程稿已吸收并清理。
- 产品：独立 Design、Optimize、Implement 和单脚本 Workflow 已实现。四模块最终默认 CPU 集合 **74 passed**；现有 Template **8 项**与 Extract **6 项**相关回归分别通过。
- 切换：根 README、方法入口、依赖、测试收集与 ignore 已切换；旧根产品和两个登记 worktree 已移除。旧 `.tooling`、仅根 `.schema_extract/` 与空 `.worktrees/` 已全部移除。
- 首次完整代码 commit：已提交 `4f7919bf531ad8b7ab49d5e371dfc8b7fea51a34`（`Refactor`），材料提交为父提交 `2c9c346`。提交时修掉新测试文件末尾多余空行，未改变行为；提交后工作区干净，未 push。
- 真实 Qwen/Agate：`vector_add-001` 已实际调用默认 Qwen 三次，选材完成、两次写作在各自 900 秒后超时，Workflow 终态 `incomplete / write_failed`；本次 GPU 提交 0 次，无未结清 job。原失败完整保留，最小材料修复正在接续；真实闭环尚未成立。

## 基线与环境

实施起始为 `main@9f95eb30122678fab855f6855a322a8f7291dd7f`。接手整合时已有 A/B 对三个模块、三个根方法与固定示例的授权改动，没有把这些当用户未知改动删除。整合者不 stage/commit/switch/reset/clean，不 push。

使用 `/home/shaohanyong.shy/.cache/kernelschema-standalone-venv/bin/python`，Python **3.13.12**，实际基座 `/usr/local/python/3.13.12`，不依赖待清理 `.tooling`。PyYAML **6.0.4**、jsonschema **4.26.0**、pytest **9.2.0**、正式 **atrex-gateway-client 0.14.37**。SDK 从官方发行索引 `http://artlab.alibaba-inc.com/1/pypi/pypi-releases` 安装；其余依赖使用既有默认索引。没有下载新 Python、修改全局/AKA 环境或 vendor 旧 SDK，host 没有为此安装 torch/Triton/Bench/Ray。

主代理在 CPU 整合阶段核对：当前工具进程没有 `AGATE_AK/AGATE_SK`，默认 `~/.atrex/config.json` 不存在。已复用用户此前明确提供且授权的凭据，准备任务专用私有配置 `/home/shaohanyong.shy/.cache/kernelschema-standalone-private/agate.json`（目录 0700、文件 0600），不改全局或 AKA。正式 SDK 默认连接解析成功，没有网络/GPU 请求。后续只需 `KS_AGATE_CONFIG` 指向该文件；配置在仓库外，不进入 Git，本文不记录认证值。

native Claude CLI 已由 A/B 只读核对为 **2.1.316**；新入口默认 `qwen3.8-max[1m]` / `max`。沿原 HOME、CLAUDE_CONFIG_DIR 和登录来源，将已有 settings.env 连接字段补到仅子进程环境；禁用 hooks/MCP/插件及写工具，认证不进入提示、产物或日志。

## 已实现的独立能力

A 完成 Design/Optimize 的本地 runner/prompt、README 和根方法；B 完成 Implement 自有 coder、正式 Agate 适配及固定示例；A/B 停写后 C 接管 Workflow、根配置、导航和清理。A/B 交接稿中的有效事实已吸收到本文，过程交接文件已删除。

Design 发现 catalog 内外完整 root/dispatch，选至多三个互补近邻，固定真实 raw/effective/from 来源、DOC 和任务绑定材料；空参考仍可设计。host 收完整文件、保存原稿、规范化新输出并严格重读/校验。Optimize 实读完整 Schema/kernel 和原反馈实物，核对绑定与原上下文；严格保持 Contract 解析值，keep/repair 不写 Schema，revise 必须有实质变化，坏稿不伪装 keep。没有源码只接受真实 Implement design_gap 及具体诊断。默认调用数 3/2，材料/请求超限具体返回，绝不截断。

Implement 固定 Task、完整 Schema、eval 和源码；同设计 coder 调用默认累计 6 次，每次新候选全案例重验。多文件完整打包，正式 SDK 延迟导入，固定 request/key、upload reservation 与 job 动作先落盘；已知 job 只原 ID 查询，完成无新增反馈不重放。缺案例/比较器/计时或已知必需缺口不能 complete，基础设施终态不交 coder 假装算法修复。Schema/kernel files 字典按 POSIX 路径记录 SHA，整体摘要是排序紧凑 ASCII JSON(files) 的 SHA-256，effective_identity 是 node→Template identity。

Workflow 仅用当前 Python、绝对路径和 `shell=False` 调三个同 CLI，没有模型、SDK、Schema parser 或第二套 evaluator。支持 Task、Schema、Schema+kernel+反馈三起点；外部先 Optimize 最多 1 次，预留本次实测决定。每版只有一个 Implement 目录，repair resume 原目录、次数不清零；有效 revise 才增版并全案例重验。每版 Optimize 按唯一结果累计最多 2 次。核对固定路径、stdout、退出码和结果，动作登记与子结果提交之间恢复不重复计费；未终结写作不重放，Implement 只交原目录恢复。

Workflow 的 last_usable 指不可变 inputs/attempt 文件；必需问题清空配对，完整重验后重建。完成必须已有本次通过全部案例和计时的实现，以及基于本次成功反馈的有效 Optimize 决定。末版可选 revise 可 complete+limit；必需修复、真实失败和未结清动作仍不完成。原输入及已交付文件每次交接/恢复核对，完成运行直接复用。异常子 stdout/exit 也保留最新结果位置，最后该小修的 3 项针对性回归通过。所有入口统一绝对 result 路径和 0/1/2。

## 固定示例材料

为让 Golden_reference 绑定真实 Git 身份，主代理先单独提交六个 Task/eval 文件：

- 提交 `2c9c346cdd0373821b633012bdfece446c766d65`，消息 `Add task`。
- `schema_workflow/examples/vector_add/task.md` 与 eval 下 reference.py、input.py、shapes.json、metadata.json、eval.json；提交后原样保留。
- FP32 vector_add，合法 `1<=N<=65536`；root 域 `1..1024`、tiled 域 `513..65536`，重叠选 root；固定案例 `1,512,513,1024,1025,4097,65536`，数字 case ID，实际 from 共享数学区/流水线区。
- B 用正式 SDK 构造完整七案例请求，但未提交；完整 Template 只读检查为 2 YAML、0 规范化变化，structure/text/from/relations/dispatch 全通过，未 skip-hardware。

此材料预备提交不替代完整代码提交，不把旧 AKA 成绩或 CPU fake 结果当本次验收。

## 实际 CPU 验证

下面命令均使用上述 venv 解释器。A 原始 13 Design + 15 Optimize、B 25 Implement CPU 已通过；C 最终默认集合包含全部这些用例及 **21 Workflow 用例**：

```sh
/home/shaohanyong.shy/.cache/kernelschema-standalone-venv/bin/python -m pytest -q
# 74 passed in 26.28s

PYTHONPATH=/home/shaohanyong.shy/KernelSchema/schema_template:/home/shaohanyong.shy/KernelSchema /home/shaohanyong.shy/.cache/kernelschema-standalone-venv/bin/python -m unittest tools.tests.test_design_resolution -v
# Ran 8 tests; OK

/home/shaohanyong.shy/.cache/kernelschema-standalone-venv/bin/python -m unittest schema_extract.tools.tests.test_standalone -v
# Ran 6 tests; OK
```

两组既有回归按 canonical import 在独立解释器执行，不混用裸 tools 缓存。未修改保护区以适配测试。

CPU 覆盖实际 DOC 读取、完整 from/共享使用者、原稿保留、Contract 漂移与 no-op 拒绝、真实反馈摘要/相对实物、完整 helper、多文件上传、坏返回计数、超时进程回收、连接 env 传递/认证不落产物、同设计修复、全 case/比较器/计时判断、已知缺口与原 key/job 恢复。Workflow 用真实 fake-CLI 子进程测试三起点、外部 keep/repair 后首建验证、无源码 design_gap、续原目录、revise 重验、累计上限、必需/可选收尾、不可变配对、完成不重跑、子结果提交后恢复不重计数、异常 stdout/exit 拒绝。

额外接口用例把实际 Implement 的固定 inputs/attempt/result 交给实际 Optimize 材料装配与绑定校验；只替换模型回答和 SDK transport，请求构造使用安装的正式 SDK，确认完整源码、七案例反馈、原 job 与观测环境实际进入 Optimize 请求。这仍是 CPU 协议验证，没有 GPU 成绩。

从另一临时 cwd 对四入口逐一执行真实 `--help` 和材料缺项调用，均正确返回 needs_input、退出 1 与一行绝对 result 路径，未启动模型/SDK网络。实际 venv 的 `importlib.util.find_spec('kernelschema')` 与 `find_spec('aka')` 均为 None。新产品没有旧运行时或 AKA 导入兜底。活动 Markdown 文件本地目标检查无缺失，保留根 `#2-schema-是什么` 锚点。

## 切换与限定清理

- 根 `pyproject.toml` 保留项目元数据、Python>=3.11、PyYAML/jsonschema 和 test extra；新增 `implement = ["atrex-gateway-client>=0.14.37"]`。去掉旧 project.scripts/包发现，`packages=[]`，只承诺 checkout 脚本。pytest 默认四新模块 tests。
- 根 README 改为实际四入口、安装、三起点、结果边界和模块导航；不保留旧 API/AKA 运行要求。保留既有保护区链接与 schema 锚点。ignore 保留 `.schema_extract/`、`/.worktrees/`、缓存/Wiki 下载，新增 `/.schema_workflow/`，移除 qualification 项。
- 已删除仅根 `kernelschema/、tests/、docs/、examples/`；模块内 tests/examples 保留。
- 原稿先保全到 [historical_draft/root.yaml](historical_draft/root.yaml)，大小 **13,490 bytes**，SHA-256 **611786083f0da2b6b9aca4a789a28f05b7a7af419b55d779370667afad54b7dd** 精确匹配。旁边说明原路径、run/attempt、复制日期、未验收/未发布事实，未改 YAML。
- 两旧 worktree 普通/未跟踪状态为空；仓库内 worktree 有 541 个 ignored 文件，全部为已知 pycache/pytest/qualification，无未知 ignored；/tmp baseline 无 ignored。两个 HEAD 均在 main 和保留分支历史中。无锁/活动引用后普通执行 `git worktree remove .worktrees/schema-workflows-v1` 与 `git worktree remove /tmp/ks-opt-20261103-ryxoz_iu/baseline`，均成功；没有 force/prune/解锁/杀进程或删除分支，未删 /tmp 父目录。
- 之后 `.tooling` 已删除；运行区清理前针对两个清理根检查进程 cwd/exe/open files 均无引用，611 个 lock-like 文件无占用；27 份正式 current/receipt 无删除根引用。`.schema_extract` 的旧固定 executor/snapshot 有只读目录，普通删除遇 PermissionError；以不跟随符号链接的目录遍历核对 lstat 与当前用户所有权，仅给授权根内 8,978 个只读旧目录补 owner 写权限后续删。删除命令退出 0，根 `.schema_extract/` 及空 `.worktrees/` 均已移除，不碰正式 `.extraction/`。
- 本目录 01–05/07–14 已吸收后删除；06 修已删链接与历史阶段表述，保留用户决定原意。保留 README、FINAL_PLAN、FINAL_REVIEW、06_decisions、EXECUTION、historical_draft/。其他旧 discussion 与正式凭据引用的历史证据均保持原样。

清理终检已完成：Git 仅登记当前 main checkout；两个旧 worktree 目标及 `.tooling`、根 `.schema_extract/`、空 `.worktrees/` 均无残留；旧根四目录的 143 个已跟踪文件被删除。活动 Markdown 链接无缺失，`git diff --check` 通过。完整代码 commit 已由主代理执行，见当前状态。保护区 `schema_extract/、EXTRACT_SCHEMA.md、schema_docs/、schema_template/、schema_wiki/、kernel_wiki/` 与固定 Task/eval 六文件的 git diff 当前为零；未访问或修改 AKA 工作区及实验变体。

## commit 后的真实运行续接

由新的 xhigh 在完整代码 commit 后创建新运行目录，使用默认 Qwen/Agate：

```sh
KS_AGATE_CONFIG=/home/shaohanyong.shy/.cache/kernelschema-standalone-private/agate.json /home/shaohanyong.shy/.cache/kernelschema-standalone-venv/bin/python schema_workflow/workflow.py --task schema_workflow/examples/vector_add/task.md --eval-task schema_workflow/examples/vector_add/eval --gpu L20D --out .schema_workflow/runs/vector_add-001
```

该目录若存在先查原状态，不拿新目录洗次数或未结清 job。当前没有新真实 job 需要恢复。后续记录实际 schema/kernel、原 request/job/response、逐 case、真实 DOC/源码读取和 Optimize 决定；出现失败继续具体排障、受影响 CPU 与真实重验、最终提交。不得把本文件或 CPU 结论代替真实主链。

SDK 0.14.37 没有旧上传签名 URL 刷新接口；已上传可续原请求，已提交只查原 job。上传进程在确认完成前丢失时需定位原 reservation，不能私存签名 URL 或盲建新 reservation。首版不支持正式 SDK 无法表达的额外比较器/原生构建，不能替换 Task 判据。真实服务结果若与已核对结构不同，保留原 response 并按证据适配，不能自动当成功。

### Issue 01 修复交接

首轮失败后，新的 xhigh 修复了写作材料缺少真实 Git 身份和可读链接的问题，详见 [ISSUE_01.md](ISSUE_01.md)。受影响 Design/Optimize CPU 为 32 passed、8 subtests passed；原 001 的三次调用和失败结论不改，修复后需要新的真实运行。

范围偏离记录：i04 启动时一次 AGENTS 文件名检索误包含 HOME，返回了 AKA_experiment 及归档内的路径名；没有打开这些文件内容、没有修改，随后检索限定到 KernelSchema。后续代理命令明确使用仓库 workdir，避免再次出现；不能把这次元数据检索表述成从未触及任何实验路径。

## 新真实运行 vector_add-001

2026-11-22 UTC，D 在 `main@4f7919bf531ad8b7ab49d5e371dfc8b7fea51a34` 执行上述原定命令；启动前确认 `.schema_workflow/runs/vector_add-001` 不存在。配置为默认 `qwen3.8-max[1m]` / `max`、`L20D`，Task/eval 与既定材料保持原字节；认证仅以 `KS_AGATE_CONFIG` 指向仓库外私有配置。

原运行和全部三次调用保存在 `.schema_workflow/runs/vector_add-001/`。Workflow 最终退出 1，`result.json` 为 `incomplete / write_failed`，`latest_attempt=design/result.json`、`last_usable=null`；Design `calls_used=3`、`schema=null`。本轮没有 GPU 提交或未结清 job；原 Workflow、Design 和两个写作 PID 在终结后均已不存在。

Design 首调用真实选材成功（返回码 0、无诊断），选择空参考集：实际读取现有目录与近邻说明后，判断数学、调度、精度和 Triton 栈不匹配，按 Task 直接设计。第二、第三次调用均成功 Read `DOC1–6`、Template README/starter 和固定 manifest，但未返回完整稿，分别在 900 秒后超时；`drafts/02.json` 与 `03.json` 保留读取路径和公开诊断 `Claude exceeded 900 seconds; process group reaped`，返回码均为 143。第三次是产品自然 repair，没有更改上限或新建目录。第三次公开最终文字仅为正在写完整集合，无 structured output，不能作为完整 Schema。

公开工具记录显示写作者反复查找硬件、Golden_reference 与提交来源，尝试读取未授权的 checkout 根/Template 文本案例被拒。主代理已安排新 xhigh 只读排查，并在本运行终结、PID 核实后接管最小材料修复；固定 Task/eval、模型、900 秒上限与原失败目录不变。此失败不代表项目完成，后续仍须取得完整 Schema、七案例真实 Implement 和本次反馈 Optimize 决定。
