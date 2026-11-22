# B22 / L20D stage-ro qualification

2026-11-21。固定 KS `c17c15d8842b8ed13dc34fa0530cf112a9d4ce82`，执行源码树 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`。本代理独立复核 B20，并通过原 `Demo.create()` / `Demo.prepare_agate()` 实际提交原六作业资格的首个 probe。**实际 L20D 在新 PID namespace 内挂载 fresh `/proc` 被拒绝；原 suite 停止，完整 K1 资格未建立，其余五个 job 均未提交。**

唯一实际 job 为 **`dv_80d527ccf029`**，原 operation / key 为 **`a0d4a9577f704048b61eb3599582f072_qualify_probe`**。新增 **1 evaluator/GPU job、0 模型调用、102454ms active**。GPU 原生时长和货币费用未由原收据提供，保持 unknown。原 parent 已无 active/fence，收费 slot 已交回 supervisor。

证据根：[B22](/tmp/schema-b22-gpu-qualification-20261121-rnzs0ew0)；完整机器交接：[b22-final-handoff.json](/tmp/schema-b22-gpu-qualification-20261121-rnzs0ew0/b22-final-handoff.json)。本代理未修改生产源码、runtime、预算实现、原 Task 判据、Schema 或 AKA；没有安装、下载、改 worker/镜像/sysctl，未做 host remount，未提交/推送/派生代理，也未访问 AKA_experiment。原 main 仅写本报告。

## 独立复核与固定材料

已读适用 AGENTS、B20 报告/cpu-handoff/cpu-source-freeze、B19/B16 原报告和 handoff、当前 K1 EXECUTION 与 stage-ro/worker/remote/channel/agate，以及 C104–106 和 V01/V06/V07/V08/原 V25 权限条款。原 [atrex-gpu-gateway skill](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/SKILL.md) 和 raw-dev/终态/cleanup 规则用于实际提交与恢复。

B20 的 11 个冻结文件逐 SHA/长度与固定 HEAD 核对通过；六个算子模块又与 B20 基点 Git 原字节核对。检查记录为 [independent-source-audit.json](/tmp/schema-b22-gpu-qualification-20261121-rnzs0ew0/independent-source-audit.json)。B20 已有 90 passed/1 skipped 及 2 项补强证据，本轮未重复全库测试；本机 fresh-proc skip 从未被当作 L20D 通过。有限复核没有发现收费前需要修改的明确 KS 缺陷。

| 边界 | 实现复核结论 |
| --- | --- |
| 完整固定 stage | snapshot 对 manifest 全文件、stage.json、remote.json，以及执行时 execution.json/frozen_inputs.json 精确核对；不仅保护六个 `.py` |
| 实际加载 | worker 从核过 hash 的 byte buffer 编译 evaluator/baseline/candidate，核 loaded/source/evaluator；上传命令/文件与原 intent 绑定，末尾再核 stage 路径/字节 |
| mount/PID/proc | 六 namespace 独立性 guard 先于任何 mount；private propagation、fresh PID 1 `/proc`、stage 自身 bind/remount-ro，再验证实际 mountinfo |
| 可写别名 | 拒绝 symlink/hardlink/特殊文件、stage 子挂载和同 backing 的另一 mount 别名；外部 scratch cwd，关闭非标准 FD，新 proc 避免旧进程视图 |
| 权限与 scratch | setpriv 清 cap 并设 NNP，worker 入场前核 Inh/Prm/Eff/Bnd/Amb 全零和 NNP=1；完整 stage 写开/创建要求 EROFS(30)，scratch 独立写对照，cache/HOME/TMPDIR 指向 scratch |
| 生命周期 | launcher 建新进程组，超时 killpg；unshare 父退出杀 namespace PID 1；原 gateway 终态/cleanup 才能结算 |

这些是对固定实现的审阅结论；**不是本 job 已取得上述全部运行证据**。profile 明确 `scope=complete_evaluator_stage`、`whole_root_readonly=false`，不承诺整个根只读，也不承诺隔离同一 Python 地址空间中的任意恶意 native code。本次失败发生在完整权限证明之前。

原 `Demo.create(new_root, parent=原Ref, parent_bundle=原Bundle, authority=原路径, runtime=本轮profile, environment=实测L20D target, remote=stage-ro)` 真正生成新材料，没有用私有脚本串联替代 driver。准备阶段原 owner 字节完全未变，没有新 owner/operation/模型/job。五份请求均经原 validate_request，hardware scope 静态通过；旧 contract/acceptance/workload/environment、inputs/timing、baseline 和原 objective 字节均保留。旧 B16 Task/Q 与失败历史没有被原位更改。

| 新固定材料 | SHA-256 |
| --- | --- |
| Task | `f051f7152b3149b1d16505b68f143ceca2154ade2d365841042f6e97b5448652` |
| Schema | `3239691d521387feb1f6b3b6bf3c5bd0316eb51c1c8c8f8441b73b9807d6fbff` |
| fixed Q | `2027432b64570e6594fb56691e72f7703295aa72599e7120985424260a043cb5` |
| Reference profile | `c5fe768d064c7327861ce47b5452212c4d55d7207b34c5b48c0cd18645ce4ede` |
| qualification Q | `4877e5312db7e681312fce78af5ad43eecd38c43bbcc2957a990425ce6ab4863` |
| 原实测 L20D target | `d4b9c90c350b4db10664ac9ee66ec118c852a03f5c9ffd955204ea89e31ddb0f` |
| 本轮 2.00/600000ms runtime | `3e1766d7fad58da4e76b77c28a296b0adcb225c04e7e56704dac45a848efc530` |

runtime 精确对象来自 B21 公开 `prepared/.../business-profile.json` 的 parameters.claude；本轮没有调用模型。工具路径/hash直接复用 B20 remote-stage-ro 原件，不猜测。原 B16 的 Python3.12.3、精确 `NVIDIA L20D        `、SM10.3、Torch2.9.0+cu130/Triton3.7.1/CUDA13.0 是既有观察；**本次 probe 未导入 torch，不能宣称重新捕获该 CUDA 环境**。

## 唯一实际 probe 与停止点

官方 SDK 为 `atrex-gateway-client 0.14.37`，解释器 `/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python`，site-packages 在该环境 `lib/python3.13/site-packages`。认证仅 `KS_AGATE_CONFIG=/tmp/schema-b14-auth-dhqevy7o/agate.json` 经官方 SDK 读取；凭据未打印或复制到源码、日志、模型材料、包或请求。

原 driver 固定资格请求为 6 jobs/1200000ms/零模型角色额度，同一原 parent；每次 POST 前由原 reserve/authorize 和 UTC/monotonic cutoff 准入。实际只有一次 POST，目标 L20D、recycle=true、replay_policy=never、dev_intent=custom_harness；随后只有原 job GET。异步在途返回 settle_only 时原 owner 时钟保持活跃，没有把等待变成免费区间。

supervisor 的暂缓首次 POST 消息到达前，此 probe 已获原 ACK；所以不能把本 Demo 写成零调用/未准入。收到指令后立即停止所有后续 POST，仅查询、结算原 probe。最终 fresh-proc 的真实失败也独立满足原 suite 的停止条件。

原 gateway terminal 为 succeeded/terminal、cleanup_pending=false、execution_quarantined=false，外层 remote exit0、stdout1147 bytes、output_truncated=false。**外层成功只表示完整失败帧被送回**，受信帧内 `agate-dev-stage-ro-v1` 的 inner exit1、stdout为空，原 stderr 为：

```text
mount: /proc: permission denied.
... _stage_ro.py, line 118, in inner
mount("-t", "proc", "-o", "nosuid,nodev,noexec", "proc", "/proc")
... returned non-zero exit status 32.
```

执行控制流已抵达 fresh-proc mount，位于 namespace/tool/snapshot/alias/private-propagation 前置检查之后；没有发出完整 worker permission proof。stage bind/remount、setpriv 后 cap/NNP/EROFS/scratch 证明、CUDA capture、baseline 72+2 调用、原80批 timing 及四负例均未运行，不能签这些能力。没有尝试 host proc fallback、普通 allclose、候选内 harness、旧 bwrap 资格或另换 key/机制。

原 terminal SHA-256：`9be5354ad6ab0c76be200e06f42d77435e176b86612054ee8ba84eaf11ce49aa`。原 request/intent/ACK/submission-start/terminal 路径与 SHA 全列于机器 handoff。最终使用原 api.open_run/qualify 恢复入口，另加 GET-only transport 硬禁止新 POST，仅消费保留的失败终态，得到 phase=done、status=blocked、recovery=none、qualification=null。[原结果](/tmp/schema-b22-gpu-qualification-20261121-rnzs0ew0/final-qualification-result.json) 的失败原因文本较泛，具体 mount 原 stderr 完整保留。

可从 `standalone-demo/` 或其 `qualification/` Bundle 导入的失败证据为 `artifacts/d427c40c2cf741b1913b6d4f3d780651/qualification-original-records.json`，SHA `6d4e11cff561063bc5b3aa885fd1565a0d87f07f47e92bce936dd1758822ae5d`；这是失败原件闭包，**不存在可选用的成功 qualification**。较早的 Demo `qualification-result.json` 保留当时 in-flight 快照，最终结果与 handoff 明确指向原恢复收口；没有私改已准入 Q 或 driver state。

两个 `/tmp` 观测摘要脚本先后遇到 state envelope 读取方式、原生浮点时间不能 JCS 编码的问题，发生于原 POST 返回后及原 settlement 完成后。修正仅作用于附加摘要；生产 API、原 job/key/request/terminal/receipt 未变，未引起第二 POST或双计费。

## 原账与门状态

原唯一 authority 仍 `/tmp/schema-b14-agate-20261121-twmkup4i/.runtime/owner.json`；parent Ref/SHA、首次24 model/16 jobs/10800000ms和roles6/8/16/0、tokens不限均未变。开始前为3 models/2 jobs/286387ms；最终为 **3 models/3 jobs/388841ms**，剩 **21 models/13 jobs/10411159ms**。本 job 只有一条 sequence1 原 receipt，evaluator_jobs1、calls0、tokens0、gpu_ms=null；全账 tokens 仍 unknown，B17 append-only erratum未被掩盖。

102269ms 是首次实际终态结算快照；原失败状态恢复收口又计185ms，故最终本请求 active102454ms。最终原 active_run/active_since=null、fences=[]。全会话累计运行时模型10次、实际GPU jobs3次；历史模型失败/timeout/未知费用继续保留。

原完整 Task 判据、原权限实质要求和 K1/K4/K6 门没有降低。B19 证明的已有namespace/自身目录只读能力没有代替本次 fresh proc；B20 CPU证据没有代替GPU资格。当前 K1完整真实资格、依赖它的KS业务链与任何AKA配对资格均不能签通过。后续是否调整执行方案属于 supervisor 的新安排，本轮没有扩展新安全平台或追加收费诊断。
