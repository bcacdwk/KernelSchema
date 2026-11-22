# B42：简化交付的阶段记录与最小公开示例

2026-11-21 UTC。**本阶段完成三个公开 vector-add 示例文件和五份既有审计文档的更新；发现 AKA 默认入口中的 planner 串行循环尚缺，主代理已采纳最小接续修正；收口又收到 B41 原环境身份为空的标准评测阻塞。** 按主代理最新指令，本任务不等 B41 终态，先交接；本报告不宣称两个目标全部完成，也不提前认定新候选成功。

模型/GPU/安装/新测试套件/提交/推送均为0；没有修改生产框架或 AKA 文件。只访问 KS 独立工作树、指定 discussion、AKA 相关源码/公开示例及精确 B40 产物。没有读取、枚举、运行或修改禁止实验仓及 archive 变体，没有 home 递归搜索，也未查看 B41 的进程、工作环境或旧账。

## 可复用的三个产品文件

位置：`/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/examples/workflows/vector_add`。三个文件目前未提交，主代理统一固定；本阶段开始的 KS HEAD 为 `907c99aa1589bddf73e26079f2304b83d68a8e51`。

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `README.md` | 5726 | `0470b20c44fd05b32a53604f4e76fab201c83ae06c96d4f3e8f2e9f4cb0da8d6` |
| `task.md` | 1237 | `3a187afbce1d0f2291e8160b2fcf569aa9203dc1808bc331b9cb0777dc729f84` |
| `root.yaml` | 9106 | `58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c` |

[root.yaml](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/examples/workflows/vector_add/root.yaml) 是 B40 模型原文逐字副本，源为 `/tmp/schema-b40-write-only-optimize-20261121/optimize/artifacts/6dcd8f97b77f4bbd8ed96a01758feff6/raw/root.yaml`。没有宿主修 Schema、缩短原文、重新判效果或把旧失败当成功。写作真实 completed/revise、diagnostics=[] 只说明产物写出。

[task.md](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/examples/workflows/vector_add/task.md) 写明原 native 接口 `Model(alpha).forward(x,residual)=x+alpha*residual`，同长度连续一维 CUDA bf16，三个数字 shape ID 0/1/2，N1024/65536/1048576、alpha1/.5/2，L20D/Triton。它是可复用的任务描述，不冒称与 B40 原输入 task 逐字一致，也不复制一份新的权威 reference。

[README.md](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/examples/workflows/vector_add/README.md) 链接已公开的 AKA native Task/实测 reference metadata 与原安装说明，给出直接实现所附 root 的命令和 design→implement→optimize→implement API 片段。使用者提供自己的路径、writing runtime 与原 AKA 配置；keep 停止，native blocked 不继续投新阶段。现有自动 driver 固定 row-softmax，说明没有把它冒充 vector-add 或任意 Task driver。

三个公开文件不包含 native receipt、完整 CLI stream、模型思维/base64原件、认证或主机私有运行配置。不附候选源码或性能成功摘要，因为本阶段尚未收到 B41 新候选终态。四步片段是可复制用法，未声称整条真实运行已完成。

## 能力缺口与最小接续

当前 KS 任意任务文件 API 已能按序调用写作与原 standard implement；固定 row-softmax driver 有有限自动链。当前 AKA `--schema --schema-path` 只是把固定 Schema 送给原多轮 coder，不会自动调用 planner。旧 `--schema-config` dynamic 可以调用 KS writing API，但实现阶段仍依赖 FixedController、RowSoftmaxTask、SchemaExecution/qualifies、原 OwnerAccount 和 finish floor，不能当作新的轻量标准 planner 循环已经可用。其写作模板是否选择 write-only 不消除这个实现路径差异。

主要代码接点：

| 位置 | 现有能力与用途 |
| --- | --- |
| AKA `src/aka/bootstrap/workflow_cli.py:80` | schema-path 选择 standard，schema-config 选择旧 launch；可在标准入口侧增加显式有限选项，保留 off |
| AKA `src/aka/workflow/schema/standard.py:111` / `:183` | 标准 composition 选择与同步 implement；复用原 native 工作流和原结果，不需换 evaluator |
| AKA `src/aka/workflow/schema/standard_plugin.py:31` | SchemaAgent 包原 light.apply，TrackedWorkflow 仅记录原 campaign 结果；这里没有 planner 迭代 |
| KS `kernelschema/api.py:291` / `:301` / `:311` | 直接 task-file design/optimize 与 schema-file standard implement |
| KS `examples/workflows/row_softmax/driver.py:346` | 已有原 native result/feedback 保留和有限阶段接线可参考，但 Task 固定，不直接挪作任意 Task driver |
| AKA `src/aka/workflow/schema/dynamic.py:102` / `:213` 与 `controller.py:33` | 旧 dynamic/floor/fixed Task 路径；不是新默认入口应继续依赖的实现后端 |

主代理已决定由新的代理继续：在 AKA 标准入口旁补一个显式有限薄入口，直接复用 write-only `api.design/optimize` 和现有 `standard.implement`，保留使用者指定 runtime、Task 与 native 设置；各实现阶段使用新 workspace，将原 native JSON 与声明的 feedback 原文送下一次 optimize，最多一次 optimize，keep 结束。不迁移旧 controller/state，不新增预算、验收、恢复平台。不热装或重启 B41 固定快照。本阶段没有为尚未实现的选项写命令，也没有擅自改框架。

## 阶段文档与已知事实

当前已推送的运行源码配对由主代理确认：KS `907c99aa1589bddf73e26079f2304b83d68a8e51` / AKA `ef8dadadd9867f298826d52ea159cf70bef93060`，AKA gitlink/release 配该 KS。B42 的新增示例未提交，不改变这份运行身份。

- `implementation/README.md`：先给当前接口、精确边界与 B30–B42 索引，再保留 B01–B13 历史表；不再以旧版本/401/GPU0/strict 前置描述当前状态。
- `implementation/delivery_audit.md`：前置当前两个目标、已知缺口、逐阶段真实结果和费用边界；原 B01–B13 审计正文按原文保留，仅调整标题层级并明确历史范围。
- `implementation/progress.json`：新增当前能力/配对/待补项和 B33/B35/B36/B40 分阶段事实。旧 units/gates 保持，旧聚合值收在 `real_model_runs.historical_rollup_through_B29_B32`，旧 open_items 保存在 `historical_open_items_before_B42`；不让旧12次模型/4jobs冒充当前总量。全会话完整 tokens/USD 保持 unknown。
- `implementation/next_steps.md`：移除 B40在途等过期说法，明确补轻量动态入口、接收 B41、固定配对后再做最终收尾；保留访问与活动运行边界。
- `01_history.md`：只在 H087 后追加 H088（B40终态）与 H089（本阶段与核心缺口）和 H090（根确认环境身份阻塞并批准停止），既有历史前缀逐字保持。

B40事实按其原报告引用：一次planner、零reviewer/GPU，40,303 tokens、native reported USD0.5819987499999999，API active393087ms，真实 completed/revise、diagnostics=[]。B33原API失败，B35/B36无新候选及原费用未知均保留；B36只读其报告，没有重新接触运行环境。

B41原声明限额2model/5jobs/2400000ms、max_iters1/handoff_resumes1/agent900s/episode2100s。最后只有V0 `ev_b58cf6b2a519` 三shapePASS，无candidate评测；它是reference baseline，不是新candidate。主代理随后确认原 `environment.identity=null` 会使standard full evaluation、原CandidateGate六seed及ABBA均不可用，已批准通过原生停止路径结束第一调用、禁止第二continuation。B42仅据主代理消息记录，没有自行查看环境。实际停止终态、slot释放与费用仍待原B41收口，不能据“已批准停止”推定已结算。正常路径通常四jobs只是预算说明。下一轮需同时补真实环境身份兼容，不伪造身份或跳过原检查。

## 有限检查与精确清单

本次只对新示例/修改文档做局部检查：root原字节与SHA、YAML mapping可读、task数字shape与原AKA示例一致、README Python AST与shell `bash -n`、当前文档链接与锚点、progress JSON可读、历史段/旧状态保留和diff空白。未执行示例、调用 API/模型/GPU，未重新评审Schema效果或运行测试套件。检查原件见 [verification.json](/tmp/schema-b42-simplified-delivery-record-20261121/verification.json)。

精确分组清单见 [change-manifest.json](/tmp/schema-b42-simplified-delivery-record-20261121/change-manifest.json)：

- **KS产品示例：** 仅上述3个新增文件，均在独立worktree，未stage/commit/push。
- **原main指定discussion：** `implementation/README.md`、`implementation/progress.json`、`implementation/next_steps.md`、`implementation/delivery_audit.md`、`01_history.md`，以及本报告 `implementation/b42_simplified_delivery_record.md`，共6文件。
- **AKA产品文件：** 0。生产Python、测试、子模块、release、原main算子提取与B41运行文件均未改。

本阶段原文副本及变更前SHA位于 [before-manifest.json](/tmp/schema-b42-simplified-delivery-record-20261121/before-manifest.json)，仅用于本地审计。公开产品只交付三个最小文件。主代理可据清单固定新示例，接着补当前缺口；最终文档需待新入口和B41真实终态后由新的代理统一收口。
