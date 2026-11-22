# B14 / real Agate validation

2026-11-21。KS 唯一实现 writer；开发工作树 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`，起点 `1c9e592c6e0ec021f91e17af5a8c39956ce39178`。在原main仅写本报告；其余改动限指定KS开发工作树与新 `/tmp` 验证根；未访问、枚举或运行 EXP，未改 AKA、原 main 提取、AGENTS、旧算子；未提交、推送或派生代理。

本有界阶段已收口，原操作已终、账户可交接。认证接线已固定于 `204cde57b3d80426bc5367ffa186f69669513f59`。唯一真实诊断 POST 得到入队前 `400 dev_cluster_disabled`，原操作已结算；未创建GPU job、未启动模型，worker/dev/readonly 资格尚未通过。历史 7 次运行时模型调用、3 个 120 秒 timeout scope、总费用 unknown、历史 GPU=0 不改写。

## 实际修复与认证

- `AgateHTTP.from_config` 复用官方 SDK 的 `AkSkAuth`、`load_config`、`resolve_profile`。标准 `AGATE_AK/AGATE_SK` 完整环境对优先于文件；半对不跨源拼接。默认配置文件可缺失，显式 `KS_AGATE_CONFIG` 必须是现有文件；命名 profile 路由、无 redirect、每逻辑 POST 零自动重试、无秘密 identity 保持。
- 官方 SDK 0.14.37 没有公开的环境鉴权 resolver，CLI 自己选择环境值。本补丁只增加这项有界配对逻辑，不复制 token 算法或建立凭据框架。
- `test_evaluation_agate.py` **22 passed / 9.70s**，包括无文件环境对、显式文件与命名路由、环境优先、显式坏路径、缺失/半对与无秘密 identity；`git diff --check` 通过。测试使用假凭据。
- 另在实际子进程内从指定私有官方配置加载用户提供的 AK/SK，只注入该子进程环境；明确不存在的默认配置路径下，`transport_from_config()` 对 H20 GET 返回 **200**。凭据没有输出、复制到证据、放入仓库/Task/模型资料。

唯一证据根：`/tmp/schema-b14-agate-20261121-twmkup4i`。官方认证文件仅按主代理指定的私有路径使用；认证方式为 SDK AkSkAuth。本文和材料不记录任何凭据。

`evidence/index.json` 保存只读 GET 路径、状态、长度、SHA256，原 body 保留为 `.raw`。`/v1/env`、`/v1/env/H20`、`/v1/env/H20/capabilities`、`/docs/api` 和 dev/env/get-job 文档全部 HTTP200；无 `--force` 或隐式探测提交。环境认证子进程原件另为 `evidence/env-auth-h20.{raw,json}`。

H20 单项为 `source=cache`，声明 NVIDIA H20 / sm_90 / torch2.9.0+cu130 / Triton3.7.1 / CUDA13.0，原404字节摘要 `4f8889f36861266c8489376a60b9747caae7218f8eda3478e8312e6bf1f13aef`。env inventory 的 H20 池 driver580.82.07 与单项580.105.08存在差异，不把alias cache当统一worker镜像；typed target仅明确需要匹配的runtime字段，实际worker仍须同原K1通道测量。

公开 env/capabilities 没有返回 dev ACL、Python/bwrap 绝对路径或 namespace 权限。官方 dev 文档要求独立授权，未提供只读 ACL 查询。不能因此断言用户无dev，也不能断言已通过。同parent唯一受信诊断随后被明确入队前拒绝，详见下节；原六作业套件未启动，没有安装bwrap或改造服务。

## 唯一 parent 与调用前预算

parent 首次创建前，主代理将范围明确为 row-softmax 的 KS 独立及 AKA fixed/dynamic 两条路径。原单K6 demo提议10模型/8jobs/3600000ms没有创建或消费；为包含诊断、六作业资格与两后端，首次创建上限为 **24 model_calls、16 evaluator_jobs、10800000ms active**，角色 planner6/coder8/reviewer16/extract0，tokens=null。角色额度不是可相加的总调用许可；总model24继续取紧。它不是改旧失败账户、事后退款或重置余额。

- bundle/root：`/tmp/schema-b14-agate-20261121-twmkup4i`
- 私有唯一 authority：该根 `.runtime/owner.json`
- public parent Ref path：`artifacts/d60b4fa1c8854d0187afefdb6368bf39/parent.json`
- public parent SHA256：`85f19bfe92d717c94950442d3fbab16e63bccf52a0f4cb8594a71c6fd02ca3a7`
- account ID：`5b6979f18703468aaa5cf9c052776d4e`
- 交接：`parent-handoff.json`。创建时 operations=0；所有本轮Q/R必须复用该parent/authority，收费GPU与模型串行。

## 材料与待执行原件

新增项目目标文档 `examples/workflows/row_softmax/references/h20-target-b14.md`，记录准确cache原文/hash、目标声明、参考来源及无worker/微基准资格边界；已以真实204cde57 Git locator封入 `target.json`，SHA256=`0d3e13f255715146ed2b9b5462a00b40c2e70d05d10562dcc3fc52ccd01a3894`。原R0在本轮/tmp `r0-h20` 副本只绑定硬件Contract，Design/Realization逐dict不变；原Template `check_directory` 含硬件检查通过。它是目标资料检查，不是实际GPU资格。

实际从GitHub contents API取回并保存了官方不可变来源：Triton v3.5.1 commit `0add68262ab0a2e33b84524346cb27cbb2787356` 的介绍，以及 NVIDIA/cuda-samples commit `0ec046d5e99d7214cc8e9fbac2cc1e66f3019878` 的 `Common/helper_cuda.h`/BSD-3-Clause LICENSE。原API响应、源字节、blob IDs/hash都在 `sources/`；不伪造cache Git commit，不把旧文档版本当远程3.7.1实测。

`diagnose_worker.py` 和 `diagnostic_owner.py` 是本根私有新诊断脚本，已在固定204cde57代码下执行一次。一次官方dev诊断，稳定原key/原ack/query/terminal；受信代码只记录Python/stack/device与既有bwrap的同profile namespace/readonly探针，临时scratch外不修改文件。请求1job、active300000ms、worker60s；未收到原job定位则unknown，不重新POST。原6-job suite之后仍完全保留probe、完整baseline和四负例判据。

## 真实诊断、结算与准确阻碍

2026-11-21，固定KS204cde57与唯一parent下执行官方 `POST /v1/jobs/dev`，目标H20，`recycle=true`、`replay_policy=never`、`dev_intent=inspect`。服务立即返回：

```json
{"error_class":"validation","reason":"dev_cluster_disabled","message":"dev execution is not enabled on any matching cluster","details":{},"trace_id":"req-48ea08c2f0c9"}
```

HTTP400，**没有job_id**。这是匹配集群未启用dev的明确入队前拒绝，不是401，也不足以证明AK的独立dev ACL通过。worker脚本没有执行，Python/bwrap/namespace/readonly仍未实测。没有更换通道、身份或原key重投。

原 operation/key：`b14_h20_diagnostic_12f2ecfd979546328edba23d02305a2a`。原件在 `.runtime/diagnostic`：request SHA256=`8727e51a710f5a31dc82ddeee0c9b46044186911f9ac273c245fd7c4b6bb1d94`；acceptance.raw SHA256=`09934591f9c54ffec0d698c52ebd80da938652a1cd37007769d703d3e676a12e`；settled.json SHA256=`38ed5aa35dcfb0033439ce59fd3da2219775c90cb5a81bf12195456c11a8101c`。

原parent吸收该明确拒绝收据：calls各0、evaluator_jobs=0、tokens=0、gpu_ms=0、active_ms=531，fences=[]，操作 terminated=true。一次逻辑POST尝试与0个实际服务jobs分别记录；不把拒绝伪造为成功执行。原始HTTP字节保留，规范化拒绝摘要是新artifact，不冒充原字节。

服务当前不支持本H20匹配集群上的必需dev入口，因此不启动原六作业资格。最小可行后续是管理员提供/启用既有正式prod中获准H20 dev的cluster/路由；或者由任务owner明确选择另一个实际允许dev的GPU并绑定新Task，同时保留本H20失败。普通 `agate eval` 默认allclose、`check`、历史不同设备成绩不能满足完整Task比较器、readonly和六作业资格。没有安装或改造服务。

## 当前增量与停止点

- 新实际模型 calls：0。
- 新实际POST尝试：1；新GPU/evaluator jobs：0。
- parent新增operation：1，原件已结算，active531ms；fences=[]，没有在途进程/job。
- 新已发生模型/GPU费用：0；HTTP GET与拒绝响应不作为worker/GPU资格。
- K1.real/K6 GPU依旧未通过；六作业probe/baseline/四反例均未启动。目标材料已真实固定。
- 模型准备基础Task/Q已用稳定公开API独立冻结，见下节。主代理另安排最小driver外部authority复用入口，真正standalone链等其独立固定SHA；本代理不改或消费活跃driver。GPU收费slot已因明确静止交回根，继续工程与模型准备不借该拒绝伪造GPU结果。

## 有界只读普通 eval / custom SOL 可行性复核

新增只读 GET `/docs/api/eval`、`/docs/api/sol`、`/sdk/api-reference/client/` 全200，原字节及hash在 evidence/index.json。另读官方skill reference-dir/custom-sol-problem/solution-json/evaluation/operations和已安装0.14.37 public payload/Client。完整逐项表在验证根 `ordinary-eval-feasibility.md`。

普通atrex eval可投递Model/reference/_make_inputs/shapes、设置atol/rtol/bench_iters、整轮ABBA；custom SOL可投递run/generate_inputs、destination-passing输出、多文件及workload tolerance。统一job/key/query与声明OSS输出亦可复用。**已检查的公开接口没有建立与K1完整合同等价的现成路径**：缺可核的受信逐调用副作用/原返回对象与storage/guard钩子、72+2固定状态轨迹、K1原80份batch样本及精确AB/BA时序、同readonly/pinned-bwrap边界、完整原始报告和实际loaded主体证明。可调数值容差或一个Model包装器不能代替这些条件。

这是对本次实际文档/SDK证据的结论，不是断言技术上永远无法添加adapter。`runner_overrides/harness/extra_fields`存在不证明服务接受所需自定义hook；没有尝试猜参数。TaskModule是需trusted AK和cluster opt-in的preview，未获当前可用资格；check也不执行完整forward。未提交额外eval/SOL/dev/validation作业、没有换GPU盲试、没有改服务或安装包。

最小接线仍是现有获准prod H20 dev route/cluster，先核实际Python/bwrap，再原六作业；若要普通eval等价替代，先取得其已支持的受信K1 harness/只读隔离/完整报告接口证据，再做明确adapter及同parent真实六作业资格，不能用默认allclose或candidate中夹带harness直接宣称合格。当前GPU阻碍准确保留为 `dev_cluster_disabled`。

## 不消费活跃 driver 的模型基础 Q

按主代理后续指示，直接用公开 `freeze_task/consume_raw/implement_request/validate_request` 冻结模型准备基础Q，未读或执行在改的B15 driver。位置：验证根 `model-input/fixed.request.json`；handoff为 `model-input-handoff.json`。

- Q SHA256：`238ce95c45441577e043640f2390944c59988679bc5ef8e3dd97a0d878379757`
- Task SHA256：`69495dab26a5432da52e5f955613348cf1869533d1472405356a690756251b05`
- R0 Schema SHA256：`133f08ba2652931d98dcd6a77cfc9c29f7b5f64d228b32e0d2521c00fe937861`
- 使用上述原parent、准确H20 target/R0及600000ms profile；evaluation absent，基础Q的jobs上限0；显式无GPU资格。模型负责人按已有规范派生运输专用objective。
- 该准备没有创建owner、注册request、预约operation或调用模型/GPU。真正standalone完整链仍须B15固定后的原driver实际运行，不能用这份准备Q代签。

## 阶段收口与交接

主代理已接受本轮只读替代结论，并安排新模型代理继续，不因GPU条件未满足停掉模型工作。完整机器交接为验证根 `b14-final-handoff.json`，汇合唯一parent/authority、固定Q/Task/R0、target、原diagnostic收据与账户终止检查、替代矩阵和报告。该交接不创建新account、不授予新余额或重新提交原请求。

原诊断已再读留存终态一次：无网络、无新POST，仍恰好1 operation/1 history row、terminated=true、active531ms、fences=[]。最后阶段核验只读本地文件，未新增网络或费用。后续若用户提供已获准dev入口，由新的GPU阶段沿同本轮parent和准确新请求推进，同时保留本H20失败原件；不能原位改旧profile或重新执行已拒绝的原operation。
