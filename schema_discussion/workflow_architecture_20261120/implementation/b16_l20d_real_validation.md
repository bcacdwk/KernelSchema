# B16 / L20D real validation

2026-11-21。用户明确选择 L20D 后，本轮沿原 B14 parent 执行了一次独立真实诊断。原 H20 失败、Task/Q、原收据与费用历史保持不变；没有重新投递 H20。KS 唯一开发树为 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`，起点 `12cdc654d5344d543be7bbe819a0472e239c4e5c`。本代理不提交、不推送、不派生代理，不访问、枚举或运行 AKA_experiment，不修改 AKA、原 main 提取或 AGENTS；原 main 仅写本报告。

**真实结果：L20D dev 诊断成功并完成清理，但未解析到现有 bwrap，namespace/readonly 探针没有执行，原六作业 K1 资格没有启动。** 新增 1 evaluator/GPU job、0 模型调用。GPU 原生时长和货币费用未由该收据提供，保持 unknown；不记为免费执行。历史 7 次运行时模型调用、3 个 120 秒 timeout scopes 和 unknown 总费用不改写。

证据根：`/tmp/schema-b16-l20d-20261121-n5tg6rp7`。认证仅通过现有官方 SDK 0.14.37、显式 `KS_AGATE_CONFIG` 私有配置读取。凭据没有输出、复制到证据、仓库、模型输入或 Request；没有安装依赖、下载解释器或修改服务。使用了 [atrex-gpu-gateway skill](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/SKILL.md) 及其 raw-dev/作业边界说明。

## 原账户与准确费用

- 原 bundle：`/tmp/schema-b14-agate-20261121-twmkup4i`。
- 唯一 authority：原 bundle 的 `.runtime/owner.json`，原位使用，没有复制账户。
- public Ref：`artifacts/d60b4fa1c8854d0187afefdb6368bf39/parent.json`，SHA-256 `85f19bfe92d717c94950442d3fbab16e63bccf52a0f4cb8594a71c6fd02ca3a7`。
- 原 ID `5b6979f18703468aaa5cf9c052776d4e`，首次固定上限 model_calls24 / evaluator_jobs16 / active10800000ms，roles planner6/coder8/reviewer16/extract0，tokens=null，deadline_utc=null。总 model24 继续取紧，换 GPU 未增加余额。

`diagnostic-account-check.json` 是 **2026-11-21 03:22:06 UTC 的 GPU 阶段结束快照**。恰好两个 operations 均 terminated，各一条原 receipt，active_run=null，fences=[]。H20 原 operation 为 `b14_h20_diagnostic_12f2ecfd979546328edba23d02305a2a`，jobs0；L20D 独立 operation 为 `b16_l20d_diagnostic_2ebb6e8cfac34b1d822cba45cb9f5860`，jobs1。模型 calls 都是0。

L20D request active 为 **64480ms**；parent 累计 active 精确值为 **65012ms**。各 request 向下取整后相加为65011ms，与原 parent 纳秒累计再取整相差1ms，报告以 parent 精确累计为准。该快照剩余 model24 / jobs15 / active10734988ms，roles仍6/8/16/0。后续模型消费归其原操作和后续快照，本报告不把本阶段快照冒充永远不变的余额。

原 H20 target、诊断 request、acceptance 和 settled 文件均重新按 B14 既定 SHA 验证，字节未变。原 parent 的 public descriptor 与 Ref 也按 Bundle 校验。没有新 owner、退款、重置或轮换路由。

## 官方只读环境资料

两个非 force GET 均 HTTP200，原字节位于 `evidence/`，索引为 `evidence/index.json`：

| 路径 | 字节 | SHA-256 |
| --- | --- | --- |
| `/v1/env/L20D` | 411 | `99b24a8a502c00864ccd8a77c086a8b518520eba37e5c8fd529cc91473d81dd6` |
| `/v1/env/L20D/capabilities` | 11936 | `fa5a8891b3fea1087e5f43dc2b119ace718b43535df53e6331e89f91f6c90392` |

两者均明确 `source=cache`。环境声明 NVIDIA L20D / sm_103 / Torch2.9.0+cu130 / Triton3.7.1 / CUDA13.0；这些资料没有被当作 worker、dev ACL 或隔离实测。没有 `--force` 或隐式新 job。

`l20d-target-b16.md` 首次由根代理提交为 `95e6df5af2f634d559208928ffad63804db99d5f`。这版缓存声明冻结为证据根 `target.json`，SHA-256 `b96fa729c4300b2cfdf4475a8a072db9b399588d2b5b0768c51e7c9d7a624b0d`，及 `r0-l20d/`；原件保留。实际诊断后另冻结精确实测名称的 target，不覆盖这份前期声明。

## 唯一实际诊断与阻碍

复用 B14 已审查的 worker 脚本原字节，SHA-256 `d35625817e29075f1a343fdd066c87dec49682f41631c732e663793b19c40296`；其 `KS_B14_DIAG_V1` 标记保持原样，并不表示重跑 H20。新 owner 入口仅调整 L20D 目标、独立 B16 purpose/key、原 parent 明确定位以及既有 `transport_from_config`。preflight 保存两脚本 hash 和边界。

一次 `POST /v1/jobs/dev`，`recycle=true`、`replay_policy=never`、`dev_intent=inspect`，worker60s / request300000ms / 最多1job。稳定 key/operation 是 `b16_l20d_diagnostic_2ebb6e8cfac34b1d822cba45cb9f5860`，请求 SHA-256 `1d9cc8ed441532c9d0ea59be61c5a33b0477ba28597e167e40599265a0ea8482`。服务 HTTP202 返回原 `job_id=dv_37ffeb8cd2f2`，trace `req-6beaabc636dc`；之后只按该 job GET，没有再次 POST、换 key 或改路由。

原 terminal SHA-256 为 `2a9ad595c4f1b4d94a54a1825a60aa67c96f6664b3589d13936a049f7656a7a5`，原文件 `.runtime/diagnostic/terminal.raw`。状态 succeeded / terminal / stopped，cleanup_pending=false，execution_quarantined=false；exit_code0、stdout464 bytes、output_truncated=false。完整帧和长度已校验，派生摘要 `diagnostic-observed.json` 明确指向原 terminal hash，不冒充原始字节。

| 实测项 | 值 |
| --- | --- |
| Python | `/venv/bin/python` → `/usr/bin/python3.12`，3.12.3 |
| Python binary SHA-256 | `1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118` |
| Device name | `"NVIDIA L20D        "`，精确保留8个尾空格 |
| Architecture | `[10,3]` |
| Stack | Torch2.9.0+cu130 / Triton3.7.1 / CUDA build13.0 |
| `shutil.which("bwrap")` | null |
| namespace / readonly | 没有执行，未建立资格 |

这次真实 dev 执行说明该请求获准并执行，不外推所有路由/能力。bwrap=null 仅说明本诊断没有解析到其路径，不能证明所有文件系统路径都不存在 bwrap。因为没有建立可用的绝对路径、二进制 hash 和 namespace/readonly 条件，原六作业 probe、完整baseline及四负例均未投递。没有安装 bwrap、追加探测 job、普通 eval/SOL 替代或服务改造。

## 目标与 standalone 准备

新目标保留原完整 comparator、FP64 golden、72次基本调用、两调用复用轨迹、四形状原时序和四个负例判据。原CUDA来源 `NVIDIA/cuda-samples@0ec046d5e99d7214cc8e9fbac2cc1e66f3019878:Common/helper_cuda.h` 确有 `(major<<4)+minor=0xa3` → Blackwell，支持SM103架构名；不声称它验证L20D产品别名。沿用的Triton v3.5.1介绍/softmax原字节与MIT许可只支持算法与编程模型，不代替Triton3.7.1/SM103实测执行。

L20D R0 副本仅绑定硬件 Contract，Design、Realization按dict与原R0相同，dispatch原字节相同。实际独立链材料使用原 B15 `Demo.create(..., parent=原Ref, parent_bundle=原Bundle, authority=原路径)`，600000ms原profile。remote=None 明确保存未建立K1执行通道的事实。准备不运行模型或GPU，不创建、复制或重设账户。

实测补充资料已由根代理提交为 **`7d5fb60cf1d6f9ef7cf63dff757250bb4f614ab1`**，当前项目目标文档 SHA-256 为 `acbfe0af0ad5710d9b8b8dd08e65ad4812ba17fd1e9f814ec399380efd900e70`。冻结器先用 `git show <真实commit>:<path>` 比对原字节，再将完整原文、hash、license及真实不可变locator嵌入target；没有伪造cache的Git来源。两次KS变更都仅为该目标文档，生产API/比较器/driver未改。

最终 standalone bundle 为证据根的 **`standalone-demo/`**；机器交接为 **`standalone-input-handoff.json`**。该 Demo 是真实原入口生成，start=fixed，固定R0的seed=None；它尚未调用run/qualify或产生业务结果。

| 固定材料 | SHA-256 |
| --- | --- |
| `target-observed.json` | `d4b9c90c350b4db10664ac9ee66ec118c852a03f5c9ffd955204ea89e31ddb0f` |
| 新L20D Task | `4162e75b3f340e13dbe27714c17ad575f37823afc23096b473ab4e1d46bb48c2` |
| 新L20D R0 Schema | `285960e80e5170b9ee5f3c77c93f849ffdcbe93cea42c49fedba3e60c7b62a97` |
| `design.request.json` | `c3e7dc2831856e90b6ecd9b621cc56b8b4c9e560ea0cb315c30a1d6b9421ba69` |
| `fixed.request.json` | `8ebbdebfb3606945e528ad7ae26a66b082f9c5fab4f335de8ed4ecea638c8e1f` |
| `partial.request.json` | `16cd2859ca086b87a76c91d2cc08b4a50060002a6c4d31ab8e096f5784b7429a` |
| `verify.request.json` | `9f55af077581e810e8ab443b986a886f48dde219f8fdc92d4d87c241e2fa8888` |
| `optimize.request.json` | `e29218bdf04a1c178010c2edc3d61691d2f4e0cca0d1a9e6b133c2eccf7f0e3c` |
| Reference profile | `f7b2319f3d5824b15a59dff2f50236c0bc57e9279ec5c9e3e550fe66fd79204a` |

使用的原600s runtime profile是 `/tmp/schema-b14-model-real/claude-code-2.1.316-qwen3.8-max-no-tools-600s.b14.json`，原文件SHA-256 `f86190b7c579afe95fede90ecc762670297293b283249db6d3d181f5533cb0f0`。没有改它的超时或profile内容。

本地校验通过：原Template `check_directory`（hardware未跳过）；target与诊断runtime字段逐值匹配；原R0 Design/Realization/dispatch保持；原parent Ref闭包与descriptor一致；五份Q均经原`validate_request`；600000ms profile准确且evaluation absent；Demo.create前后私有authority原字节完全一致，Demo没有本地`.runtime/owner.json`。这次准备新增owner/operation/model/job均0。只做与本次文档及静态输入相关的验证，没有重复全库测试或宣称CPU检查能证明GPU资格。

## 阶段交接

固定材料已交根代理及 `/root/b17_model_execution`，GPU收费slot已交回根。根明确安排后续新阶段研究缺bwrap时如何保留原验收边界，本代理不延伸架构、不补投GPU、不自己运行模型。完整收口交接为证据根 `b16-final-handoff.json`，索引保留原parent、静态Q/Task/R0、原diagnostic receipt、GPU结束快照、报告与源文件hash。

本阶段结论仍是：**实际L20D dev/栈观察成立；K1执行通道资格未建立。** 最小后续条件是提供现有可验证的隔离入口及原判据所需证据，再由新授权阶段沿同一parent继续；不可用默认allclose、把受信harness塞入candidate或改服务代替原资格。该后续不重置本轮1job费用，也不把此前H20入队前拒绝改写成成功。
