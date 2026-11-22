# 实施与真实运行记录

实施依据：[FINAL_PLAN.md](FINAL_PLAN.md)。用户已授权计划定稿后实施、限定清理、commit，并持续排查到三个独立功能和 Workflow 真实跑通。

## 当前状态

- 计划：max 调研与具体计划、high 精简、xhigh 终审已完成，过程稿已吸收并清理。
- 产品：独立 Design、Optimize、Implement 和单脚本 Workflow 已实现。首次实现的四模块默认 CPU 集合为 **74 passed**；交稿与环境材料修复后为 **93 passed、8 subtests passed**；004 后阅读证据与等待修复为 **97 passed、70 subtests passed**。现有 Template **8 项**与 Extract **6 项**相关回归分别通过。
- 切换：根 README、方法入口、依赖、测试收集与 ignore 已切换；旧根产品和两个登记 worktree 已移除。旧 `.tooling`、仅根 `.schema_extract/` 与空 `.worktrees/` 已全部移除。
- 首次完整代码 commit：已提交 `4f7919bf531ad8b7ab49d5e371dfc8b7fea51a34`（`Refactor`），材料提交为父提交 `2c9c346`。提交时修掉新测试文件末尾多余空行，未改变行为；提交后工作区干净，未 push。
- 真实 Qwen/Agate：001–003 均在各自已固定的 900 秒上限后终结为 `incomplete/write_failed`，原失败和预算保持。004 使用明确的历史模型稿续写材料及 3600 秒上限，write 1775 秒真正交出完整三文件，repair 508 秒完成长度修复；旧 host 误将实际 Grep 正文查阅视为未读 DOC，故仍保留失败。修后完整稿另经只读 Template 的 structure/text/from/relations/dispatch 检查通过，尚需正常 Design 交付。累计真实 Qwen **13 次**（四个业务 run 各3，另1小探针），候选 GPU **0 次**，无未结清 job；另有先前1次只读 Agate 环境查询。阅读证据与默认等待修复后将用正常入口继续，完整闭环尚未成立。

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

### Issue 02 / 03：交稿与环境资料

- 002 再次超时后，由新的 i07 实现限定候选目录内的 native Write/Edit，最终小 JSON 只列文件和决定；保留部分稿用于具体修复，无新稿的失败不盲重试。细节见 [ISSUE_02](ISSUE_02.md)。
- 唯一真实交稿/权限探针使用原 Qwen/max，43.874 秒完成；候选 Write/Edit 成功，假 input 和候选子目录两次写入实际被拒，短结构化结果正常。该探针不是 Schema 业务或 GPU 验证，进程已退出。
- 环境资料已单独提交 `51f6b2399d29bafbeacf401de45afa7011160dc8`（`Add environment`）。i08 将其通过 Workflow 的 `--reference` 交初始 Design，并补示例的实际软件版本与来源；固定 Task/eval 六文件和 Golden_reference 原身份保持不变。细节见 [ISSUE_03](ISSUE_03.md)。
- 合并两修复后，在独立 venv 新进程执行 `python -m pytest -q`：**93 passed、8 subtests passed in 39.82s**。示例完整 Template 只读检查仍通过，2 YAML、0 would change，未 skip-hardware。没有再扩大无关测试。
- 截至本次修复提交前，真实 Qwen 业务调用为 001/002 各 3 次，另有上述 1 次小探针；候选 GPU 评测仍为 0。另有 1 次只读 Agate 环境查询，不能混算为候选实测。003 将在修复提交后从新目录开始，原失败记录保留。

## 新真实运行 vector_add-001

2026-11-22 UTC，D 在 `main@4f7919bf531ad8b7ab49d5e371dfc8b7fea51a34` 执行上述原定命令；启动前确认 `.schema_workflow/runs/vector_add-001` 不存在。配置为默认 `qwen3.8-max[1m]` / `max`、`L20D`，Task/eval 与既定材料保持原字节；认证仅以 `KS_AGATE_CONFIG` 指向仓库外私有配置。

原运行和全部三次调用保存在 `.schema_workflow/runs/vector_add-001/`。Workflow 最终退出 1，`result.json` 为 `incomplete / write_failed`，`latest_attempt=design/result.json`、`last_usable=null`；Design `calls_used=3`、`schema=null`。本轮没有 GPU 提交或未结清 job；原 Workflow、Design 和两个写作 PID 在终结后均已不存在。

Design 首调用真实选材成功（返回码 0、无诊断），选择空参考集：实际读取现有目录与近邻说明后，判断数学、调度、精度和 Triton 栈不匹配，按 Task 直接设计。第二、第三次调用均成功 Read `DOC1–6`、Template README/starter 和固定 manifest，但未返回完整稿，分别在 900 秒后超时；`drafts/02.json` 与 `03.json` 保留读取路径和公开诊断 `Claude exceeded 900 seconds; process group reaped`，返回码均为 143。第三次是产品自然 repair，没有更改上限或新建目录。第三次公开最终文字仅为正在写完整集合，无 structured output，不能作为完整 Schema。

公开工具记录显示写作者反复查找硬件、Golden_reference 与提交来源，尝试读取未授权的 checkout 根/Template 文本案例被拒。主代理已安排新 xhigh 只读排查，并在本运行终结、PID 核实后接管最小材料修复；固定 Task/eval、模型、900 秒上限与原失败目录不变。此失败不代表项目完成，后续仍须取得完整 Schema、七案例真实 Implement 和本次反馈 Optimize 决定。


## 材料修复后的真实运行 vector_add-002

2026-11-22 UTC，新的 xhigh 在 `main@c352c6bd6f2bb1c6ab686dd8c0fc3562642202a1` 启动。启动前工作区干净，确认 `.schema_workflow/runs/vector_add-002` 不存在；新运行理由是已提交的材料交付修复，原 001 的调用和失败结果保持不变。

```sh
KS_AGATE_CONFIG=/home/shaohanyong.shy/.cache/kernelschema-standalone-private/agate.json /home/shaohanyong.shy/.cache/kernelschema-standalone-venv/bin/python schema_workflow/workflow.py --task schema_workflow/examples/vector_add/task.md --eval-task schema_workflow/examples/vector_add/eval --gpu L20D --out .schema_workflow/runs/vector_add-002
```

默认 `qwen3.8-max[1m]` / `max`、`L20D`、单次 900 秒上限均未改变。初始 Workflow PID `652702`、Design PID `652741`、选材 CLI PID `652990`，工具 session `92185`。该运行已自然终结，结果见下文；中间 `interrupted` 为运行中占位，最终结论依据原进程退出及终态文件。

本次选材调用成功返回（`drafts/01.json`，退出 0、无诊断），实际 17 条 Read/Glob/Grep 均完成，读取 catalog 与近邻后选择空参考集。Task/eval 六文件的 manifest Git 状态全部为 verified，reference blob 仍为 `d8c4d059869ef22c7f71e1e6f9756f94a0a02bb8`。

第二次 write 调用（`drafts/02.json`，CLI PID `656013`，session `b30eecb5-2b97-4876-9cce-6b217aea912f`）再次运行 900 秒后退出 143，公开诊断为 `Claude did not finish a successful response` 与 `Claude exceeded 900 seconds; process group reaped`；最终回答为空、结构化稿为 null。19 条公开 Read/Grep 记录全部 completed：DOC1–6、Template README/starter、manifest 以及新增 QSA 格式案例的 README/dispatch/root 前 120 行都实际可读；其余读取 catalog 并搜索 L20、Triton、SM/架构名及硬件字段，没有工具拒绝。原写作 PID 已确认不存在。材料修复解决了已见来源和访问缺口，但本次证据不能说明它们解释全部超时；仍没有可验收 Schema。

原流程随后自然进入 repair 第三调用（CLI PID `670838`），上限与计数不变。当时无 GPU 提交；重复超时立即报告主代理，新的 xhigh 在运行期间只读排查，未修改在途输入或产品。


第三次 repair（`drafts/03.json`，session `621f09e8-33ee-44e9-9810-5b655775f878`）同样在 900 秒后退出 143，诊断与第二次相同，最终回答为空、结构化稿为 null。14 条 Read 均 completed：starter、QSA 格式案例 root 全文/README/dispatch、DOC1–6、manifest，以及已有近邻说明 `materials/0023/README.md`、`0029/README.md`、`0037/README.md`；没有工具拒绝。没有完整稿可进入 Template 候选验收，不能从读取成功推出设计成功。

原 Workflow 自然退出 **1**，终态 `result.json` 为 **incomplete / write_failed**、`latest_attempt=design/result.json`、`last_usable=null`；Design 为 `calls_used=3/schema=null`。运行中没有 `vNNN`、request 或 response 文件，**候选 GPU 评测为 0 次**，无未结清 GPU job。Workflow/Design/三个 CLI 的 PID `652702/652741/652990/656013/670838` 全部不存在，另按 PID/PPID/PGID 核对也没有残留进程。原 001 与 002 的次数、失败结论和文件保持原样。

同阶段主代理曾通过正式 SDK 执行一次只读 `get_env('L20D')` 网络查询，服务缓存环境记录位于 `.schema_workflow/environment/l20d-service-env.json`，`source=cache`、`probed_at=2026-11-22T11:48:31.084555+00:00`，含 SM103、Triton 3.7.1 等信息。这是服务缓存的环境证据，不是候选实测，没有创建 GPU job，也没有回填或改变 002 的固定材料；本阶段不能表述成从未访问 Agate 网络。

终态与全部进程核实后，已直接交接给新的 xhigh `i07_design_delivery_fix` 接续产品修复；本运行负责人停止写入，不 stage/commit/push。002 只能证明来源与格式路径修复后的实际可读性及重复交稿超时，不能证明唯一根因，Design/Implement/Optimize/Workflow 真实主链仍未成立。


## native 交稿与环境修复后的真实运行 vector_add-003

2026-11-22 UTC，新的 xhigh 在 `main@ad8255b670948576e3617c2ff0ef75390849c5e6` 启动；启动前工作区干净且确认新目录不存在。新运行对应已提交的 native 候选 Write/Edit 和环境资料透传修复，原 001/002 失败目录与次数保持不变。

```sh
KS_AGATE_CONFIG=/home/shaohanyong.shy/.cache/kernelschema-standalone-private/agate.json /home/shaohanyong.shy/.cache/kernelschema-standalone-venv/bin/python schema_workflow/workflow.py --task schema_workflow/examples/vector_add/task.md --eval-task schema_workflow/examples/vector_add/eval --reference schema_workflow/examples/vector_add/environment.json --gpu L20D --out .schema_workflow/runs/vector_add-003
```

默认 `qwen3.8-max[1m]` / `max`、单次 900 秒、`L20D` 未改；固定 Task/eval 六文件未改，环境资料以独立参考原字节进入初始 Design。初始 Workflow PID `711215`、Design PID `711256`、选材 CLI PID `711522`，工具 session `80847`。当前为在途记录，`interrupted` 仅是子模块运行中占位，不是终态；候选 Schema、GPU 成绩和完整闭环均须按后续实际产物判断。

首选材真实调用耗时 **130.873 秒**，`design/drafts/01.json` 为 `state=finished/returncode=0`、`diagnostics=[]`。模型选择空设计参考集；公开 Read 记录确认本次 `input/materials/0022/environment.json` 读取 completed，环境资料不只是被列在路径中。随后自然进入写作 2/3，CLI PID `713843`，候选目录 `design/drafts/02/`；尚未交付的候选不作为完整 Schema 或闭环证据。

第二次 write 在 **900.158 秒**后退出 **143**，公开诊断为 `Claude did not finish a successful response` 与 `Claude exceeded 900 seconds; process group reaped`。但超时前已有一条真实 `Write completed`，产生 `design/drafts/02/root.yaml`，**7,468 bytes**、SHA-256 `1263810881bf9dbe7292a9d3ad4be7761c58cc7e28bd984171e6bf7836613d60`；13 条 Read/Grep 全 completed，`errors=[]`，最终短文字说接下来写第二设计，`structured_output=null`。这是实际业务部分稿，不是完整 Schema；当时缺第二节点、dispatch 和成功终态。主代理只读核对 PID `713843` 的参数，确认 Write/Edit、候选目录权限和小 JSON（文件名字符串）协议均已接入，不把本次超时归为仍在使用旧交稿方式。

原写作 PID `713843` 已退出。产品按已有部分稿自然复制到 `design/drafts/03/root.yaml`，继续 repair 3/3，CLI PID `796478`；两份 root 的起始字节一致。它没有在无稿状态盲重试，也没有清零次数；候选 GPU 提交仍为 0，后续结果仍待原调用终结。

第三次 repair 在 **900.178 秒**后退出 **143**，诊断仍为 `Claude did not finish a successful response` 与 `Claude exceeded 900 seconds; process group reaped`。公开 session 为 `48b391cf-4267-4266-926f-38c6ed1cb7ec`；16 条 Read/Glob 全 completed，其中实际读取了 `drafts/03/root.yaml`，以及 DOC1–6、Template README/starter、格式案例的 dispatch/README/root 前 120 行和 manifest。`writes=[]`、`errors=[]`，`final_answer` 为空、`structured_output=null`；原稿与复制稿的字节数和 SHA-256 仍完全一致。原第二次 write session 为 `d974c424-73b4-4f68-9cb3-02494de57be2`。这能证明部分稿被真正续读，不能证明补完或验收成功，也不能从公开记录确定超时的唯一根因。

**最终结果（2026-11-22 13:25 UTC）：** 原 Workflow 自然退出 **1**，`result.json` 为 **incomplete / write_failed**、`last_usable=null`、`latest_attempt=design/result.json`；Design 为 `calls_used=3/schema=null`。`state.json` 的 `versions=[]`、`action=null`、`verified_decision=null`。没有完整 Schema 交付，没有 kernel、Implement feedback、7 cases 正确性、计时或 Optimize 决定，不能宣称四功能闭环完成。运行目录无 `vNNN` 或 request/response/feedback 文件，**候选 GPU 提交 0 次，无未结清 GPU job**。

Workflow/Design/三个 CLI 的 PID **711215 / 711256 / 711522 / 713843 / 796478** 均已不存在；按这组 PID、PPID、PGID 再核对，没有相关残留进程。工具 session `80847` 已正常结束。固定 Task/eval 六文件逐个与 `2c9c346cdd0373821b633012bdfece446c766d65` 的原字节比较一致；工作区仅本文变更，未改产品、保护区或原 001/002，未 stage/commit/push。

已向主代理提交本次具体诊断、候选位置、调用/session/PID 与终态，交由新的修复者接续；本轮负责人停止写入，等待主代理安排下一次已修复代码的验证。003 只确认 native 部分稿写入与续读可用，完整 Design 和后续真实主链仍未成立；不得用新目录抹去这三次调用或把保留的根节点稿作为已验收 Schema。


## 003 后的有界续写验证 vector_add-004

接手者只读复核 003 原稿、公开调用及当前 prompt/runner 后确认：native 交稿和材料均已接入，host 自行设定的 `communicate(timeout=900)` 会终止整组进程，下一次 repair 为冷启动。003 根节点确实已有三层和 REALIZATION，并非零进展；没有证据把超时继续归给权限或来源。

对 003 原稿执行 Template `--read-only`，发现 Input.x/y 的 flow mapping 说明含未引用逗号，分别误生成一个 null 额外字段；dispatch 缺失，后续 relations 因结构错误尚未检查。原字节没有改动。

下一次使用现有正常 Workflow 入口，显式 `--timeout-s 3600`，以 `.schema_workflow/materials/vector_add-003-continuation.md` 作为额外普通参考。该文件明确记录原 003 来源、失败、字节数/SHA，逐字节嵌入真正模型产生的 7,468 字节根稿并附上述真实诊断；它明确是未验收续写材料，不是完整 Schema 或手写目标答案。仍附原 environment.json，固定 Task/eval、默认 Qwen/max、模块调用额不变。该次有区分力地检验更完整单次时间与具体续写是否能结束，不再机械重做从零的 900 秒调用。原 001–003 全部结果与次数保留。

运行命令：

```sh
KS_AGATE_CONFIG=/home/shaohanyong.shy/.cache/kernelschema-standalone-private/agate.json /home/shaohanyong.shy/.cache/kernelschema-standalone-venv/bin/python schema_workflow/workflow.py --task schema_workflow/examples/vector_add/task.md --eval-task schema_workflow/examples/vector_add/eval --reference schema_workflow/examples/vector_add/environment.json --reference .schema_workflow/materials/vector_add-003-continuation.md --gpu L20D --timeout-s 3600 --out .schema_workflow/runs/vector_add-004
```

该节暂记录执行选择；成功、调用、GPU/job 与最终闭环必须以本次实际结果补记。

等待 004 原调用期间，新的 xhigh 只读审查实际固定材料和 runner；未改在途输入、未新增模型/GPU。按 frozen input 与执行路径在内存重建 write 请求为 68,362 bytes、11 个 COMPLETE FILE，continuation 全文出现一次，内含原 root 7,468 bytes/SHA 精确一致，两处具体诊断在最终 WRITE 指令前。0022 环境与 0023 续稿均列在 write inline_reads。select 的“未作为参考”文字只记 notes，没有进入 write；`absent_from_commit` 只限制已提交来源声明，没有排除续稿。普通方法要求自检与自动化由 host 检查之间有措辞张力，但当前指令有明确自动化特例，不能据此认定检查死锁；全部 Wiki 绑定也不是落盘草稿的统一前置门槛。

公开事件目前只在 `communicate` 返回后解析保存，因此启动记录本身不能区分正在阅读、分析或等待服务响应。若 3600 秒仍不完成，优先检验一个更具体的首个写作检查点（先保存历史 root 并修两处已知格式错误，再补完整集合），而不是再次原样加时；这是待根据终态选择的有区分力后续，不是已证实根因。

004 在 write 启动记录后 1,657.826 秒首次落盘 root（具体时差按启动记录与文件 mtime；随后终态记录会覆盖 draft JSON mtime），当前仍是未验收候选：7,696 bytes、SHA-256 `13a26bb6c86024b0996551205dd318b4a13e82e1fc05b1d8d995776ba397e661`，mtime `2026-11-22T14:01:38.583730+00:00`。实际对照显示模型已修正两处 flow mapping 的逗号误解析，并更新若干设计/路由表述；没有人工回填目标 Schema。后续第二节点、dispatch、final 与 host 检查仍待同一调用继续。首次可见产出发生在旧 900 秒上限之后，支持此前时间截断确实会丢失这次后续进展，但尚不能据此宣称完整成功。

004 write 在 **1,775.020 秒**后以 native 成功终态返回：returncode=0、diagnostics=[]、三次 Write completed、最终 filenames/notes/missing JSON 正常。实际产出 root.yaml（7,696 bytes）、multi_tile.yaml（3,798 bytes）及中文 dispatch.md（4,315 bytes）；两个设计和显式重叠选路完整交稿，Math_reshape 真实 `{from: root}`，没有 GPU 或性能声称。公开 terminal 报告 20 turns、outputTokens=61,250、duration_api_ms=1,771,562；这里只保留用量/状态，不读取或发布思维正文。该结果直接证明 900 秒不够本次默认 Qwen/max 的完整写作，不能再归结为交稿权限或完全缺少素材。

host 随即正常规范化和严格检查，报告 **13 处文本长度超限**（meaning 50、Feature 100、Strategy/describe 150、Thesis 300 字符界限），因此尚未接受 Schema。具体诊断和三份原始候选保存在 design/drafts/02.json 与 02/，没有人工裁短或放松 Template。产品自然复制全部候选进入 repair 3/3，CLI PID `838937`，原调用总额与 3600 秒上限保持。当前累计真实 Qwen 13 次（包含在途 repair），候选 GPU 仍 0。

004 repair 在 **508.348 秒**后也以 native 成功终态返回（returncode=0、diagnostics=[]），10 次 Edit 均 completed，最终仍列出完整三文件。root 缩为 7,049 bytes，multi_tile 缩为 3,483 bytes，dispatch 原 4,315 bytes 保持。它实际 Read 候选，并使用 `Grep(output_mode=content)` 对固定 DOC5 的 `meaning|describe` 与 DOC1 的 `range|meaning` 查阅相关正文；旧 host 只把工具名 Read 计为 DOC 阅读，在进入候选验收前拒绝：`Writer must actually Read the relevant DOC1–6 material before deciding`。这是已定位的工具证据判断缺陷，不是模型超时或缺少全部设计文件。

**004 最终**：Workflow 退出 1、`incomplete / write_failed`、last_usable=null；Design calls_used=3/schema=null。session `50998` 正常结束，三个 CLI PID `813311 / 815756 / 838937` 全部不存在。候选 GPU 请求 0，没有未结清 job。原目录、失败与计数保持不改。累计真实 Qwen 13 次（001–004 各3次，另1次小探针），另有先前1次只读环境网络查询。

随后对完整 `drafts/03/root.yaml` 执行正式 Template `--read-only`：structure、text budgets、from、relations 和 dispatch 全部通过，2 YAML 仍需正常规范化预览（CLI 因 2 would change 返回1）；没有写回原稿或将本项只读检查冒充 Design 已接受。新的 xhigh 专门接管 D/O 正文阅读证据修复，并将四入口默认单次等待统一为可覆盖的3600秒；修复后的正常新运行将使用这份真正模型产出的完整未验收稿继续，避免重做旧的从零写作。

Issue 04 修复由新 xhigh 完成并交接，D/O 正文匹配只保存精确路径/命中行数，四入口新运行默认3600秒；旧run配置/结果不变。必要 CPU 为 **97 passed、70 subtests passed in 55.30s**，最后补充正文内定位符歧义后，仅定向重跑为 **4 passed、64 subtests passed in 1.25s**。另已用模型/SDK替身确认默认值、正有限覆盖、非法值拒绝与原900秒resume不变。旧004 JSON只保留Grep请求及completed状态，并不包含实际返回正文；固定DOC存在对应模式匹配（DOC5为53行、DOC1为27行），不得事后伪造旧调用的content_hits。保护区与固定Task/eval的git diff为零；git diff --check通过。
