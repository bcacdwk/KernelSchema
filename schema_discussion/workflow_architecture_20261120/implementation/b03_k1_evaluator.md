# B03 / K1.evaluator 与 K1.real 交付

2026-11-20。源码仅写 `KernelSchema/.worktrees/schema-workflows-v1`，分支
`feat/schema-workflows-v1`；基于 core `c1d7bd9cfe941e7440e6f3f8a2c6ae8d9b20bf52`、
K1 材料 `35443d2f6cefe424e9adcd6d8d66dc17aa267d99`。未提交、推送或修改 Git 配置。
未访问、修改或运行 AKA_experiment；AKA 仅只读现成 gateway skill、文档和相关实现字段。
没有模型调用、正式优化实验、GPU job 或原生 V0/晋升。

**K1.evaluator：实现及适用 CPU 检查已交付，可供 K4 消费，待主代理签收。
K1.real：未运行，未取得真实资格。** 配置未在本会话找到与远端技术能力未验证是两件独立事实。
本报告不签 R0 环境准入、CUDA/Agate/AKA 支持、K4 fidelity 或产品 gate。

## 实现与实际接口

唯一使用说明：源码树 `examples/workflows/row_softmax/evaluator/EXECUTION.md`。

- `kernelschema.evaluation.task.freeze_task(Publication, environment=Ref)` 发布真实 JCS Task/evaluator、baseline/starter Source。evaluator 完整文件集包含 golden/baseline/input/check/timing/entry 及实际 `_worker.py/_sandbox.py/_controller.py/_remote.py` Ref；执行代码改变也改变 evaluator 身份。
- `stage(Bundle, task_ref, source_ref, new_directory, scope=scope_ref)` 核验实际 Task/输入/计时判据及全部实现字节，复制准确 single_file `kernel.py`，无工作区 overlay、旧 import 或返回值修复。worker 在新进程内直接编译刚校验的字节缓冲；回执核 actual-loaded map、C/evaluator/环境/冻结输入摘要。K4 必须传所选 scope；部分 scope 在创建 stage 前拒绝。独立调用省略 scope 明确表示全 Task 评测。
- `RowSoftmaxTask(bundle, task_ref).validate_scope(schema_ref, scope_ref, require_hardware_binding=False)` 复用实际 Template resolver，检查 Task 的四个 coverage 分区、绑定节点及 root Contract 继承；仅解释分号连接的 `m/n = 数字`、`m/n in {数字}`，root 无 when 表示 Task 全域。明确排除 fail，未知 prose/non-root 无条件/variant 返回 unknown。`require_hardware_binding=True` 仅检查可识别的硬件文档字段/placeholder：具体非 pending 绑定可通过，当前 R0 仍 unknown；不验证引用文档真实性或设备/profile 资格。真实执行资格由执行器和 K4 组合，不再硬编码进静态验证器。
- `evaluation_capabilities()` 明确返回 `scope_mode=full_task_only`、四个 coverage_ids、`subset_only=false`；`require_evaluation_scope(scope_ref)` 对部分 scope 抛 `UnsupportedError`。validate_scope 和显式传 scope 的 stage 均调用它。当前 evaluator 始终跑四 shape、36 case/72 basic call、两次复用轨迹和全四 shape timing，不支持 subset-only，不能将静态 scope pass 当成运行资格。
- `LocalExecutor(root, python=...).prepare/locator/submit/query/settle` 是可实际运行的 CPU logic 通道。调用者提供唯一 owner、原 operation/key 和 `{owner,evaluator_jobs:1,active_ms:...}` 预约；locator 可先写入 K2 RunContext intent。prepare 不导入 candidate，先冻结实际 CPU FP32 输入及环境。submit 先持久化意图，再启动独立 controller；caller 断连后原 controller 仍保存原终态。
- `AgateHTTP` 消费官方 SDK 的 `Connection/AuthProvider`，使用文档化 HTTP 路径，保留响应原始字节，禁重定向泄露鉴权、禁自动 POST retry/legacy fallback。`AgateDevExecutor(root, transport).prepare/locator/submit/query/settle` 构造真实 dev wire：准确 files、固定 GPU、稳定 idempotency_key、`recycle=true`、`replay_policy=never`、不超过预约的 timeout。它不是普通 eval 的 allclose 适配。

核心及 evaluation 包 import 不加载 torch/CUDA。未改 contracts/api/artifacts、共同 fixtures/conftest、R0、Task 数字、runtime/claude 或 pyproject。

## 完整判据与计时

input.py 按唯一 inputs.json 生成四 shapes×九实例，锁定 torch CPU Generator 和实际 FP32 字节。
每调用前保存 x 的 FP32 CPU 副本与完整 backing 原字节；未改 golden.py 的 CPU binary64 公式。
check.py 跑全部 **72 次基本调用**，并另外复用同一个 Model/x storage/out storage 做 **两次变化输入轨迹**。
8 次 seed23 poison 调用实际保持 x offset=3、out offset=5。

返回对象身份、原 out storage/view、写满、x 对象/storage/view/字节、out 前后 guard、finite 与逐元素容差分别出结果。
保留活的原 storage/backing，从原 out backing 取 actual；同 out 对象 `set_` 换 storage 仍失败。
golden 的输入只能来自 pre-call 副本，候选改 x 后不能污染 oracle。原 output/golden 字节和布局/guard 原后值都保留。
host 从不为 candidate 写 out/copy 返回值。有限 canary/offset 样本不证明所有设备内存安全。

timing.py 每 shape 两方同输入、独立预分配 out，单列构造/首次调用/20 warmup；每批同步→100 次完整 forward→同步，
10 对 AB/BA 顺序固定，保存 **80 条**有序原始 ns 样本及 median/IQR/min/max/count。
绝对 perf_counter_ns 超 JCS 安全整数范围时保存精确十进制字符串；elapsed 保持整数，不转浮点丢精度。
前置 correctness/构造工作明确计入 preparation 说明，不能冒称计时区首次调用一定是 cold compilation。
软目标独立于完整测量状态，确定性 CPU 时钟用例证明 `complete=true` 可同时 `soft_goal=unmet`。
CPU 样本只验执行逻辑，不能代签 CUDA timing 或优化收益。

## 本地权限与原作业结算

本机 Linux x86_64 seccomp TSYNC/no_new_privs 只读过滤器在候选 import 前生效，覆盖所有线程；实际逐文件写打开探针以及候选写 golden.py 均返回权限错误。
它禁止文件写/元数据变更、进程/网络创建等，也禁止 Triton 编译缓存写入，故 **只声明 CPU logic，不声明 CUDA 支持**。
这证明文件系统写边界，不声称恶意本地机器码无法修改同进程 Python evaluator 内存。

Local controller 保存原 stdout/stderr、退出码/timeout/原 source 与 stage/execution 身份；查询只认原意图与真实原终态。
提交中断/ACK 丢失保持 unknown；同操作重复 submit 只 query；未知操作 fence 阻止新 operation/key。
甚至 intent 已写但 Popen 未执行的不可证明边界也不会重提。未知费用不补零，不造第二套账本。
settle 用原 publication ID 恢复，保存规范 evaluation、actual environment、输入/样本、原 receipts 及 usage_index。
同 producer+operation_id/receipt_id/sequence 重放返回原 Ref、`new_receipt=false`。
父级 active clock、跨通道 fence、预约和累计账目始终由调用者管理，不把 CPU/服务 wall time 另算一份父时间。

## Agate：配置事实与技术能力分别记录

实际采用技能：
`/home/shaohanyong.shy/atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/SKILL.md`，v0.15.0，
以及同目录 reference 的 operations/jobs/reference-dir/evaluation 文档。还核对官方安装客户端 dev/HTTP 字段。
用户授权后，在自己的隔离环境安装官方 **atrex-gateway-client 0.14.37，wheel 约 94 KB**，没有改系统/他人 venv。

配置事实：本工具 HOME 实际为 `/home/shaohanyong.shy`；显式该 home 的 `.atrex/config.json` 不存在，shell 无 AGATE/ATREX_SANDBOX 鉴权配置。
`.claude/skills` 不存在，`.codex/skills` 只有 `.system`，未找到另一个 bench 技能入口。
实际默认 prod `agate env` 已连通服务，但返回 **401：缺少鉴权头 Access-Key, Timestamp, Token**。
没有输出密钥、改 HOME、广搜 history/EXP 或自行申请权限。该事实只能说明“本会话未找到现有配置”，不能推断用户已配置服务没有 GPU 或 dev。

技术条件：实际完整通道为 **条件性的 Agate dev+bwrap**，不是已资格服务能力。
probe 只运行受信环境/输入生成并记录已有 bwrap 二进制摘要；execute 消费该原件，验证固定 bwrap，
使用已有 readonly-root + writable compile scratch + GPU device nodes 的 bwrap profile，然后调用同一个 worker/evaluator。
没有安装 bwrap、改镜像或降级比较器。远端 dev ACL、server never-replay、bwrap/namespace/CUDA、driver/torch/Triton/image、完整 stdout 容量及清理终态均 **未实测**。
完整结果经压缩封包带字节数与 SHA256，截断/缺项/错主体不能变 pass。原 HTTP/native 字节保留。

SDK 的公开接口没有 **只读按 idempotency_key 反查 job_id**。因此 ACK 完全丢失、无任何原 job_id 时，保留 key 并返回 settle_only，绝不重新 POST 试图找回。
已有原 job_id 的 404、取消已接受但 cleanup_pending、quarantined 或 transport error 都继续 query 原 job；迟到原终态只吸收一次。
公开原收据没有本实现已确认的累计 GPU 毫秒字段时 `gpu_ms=null`，不能用 wall time 伪造。

**真实资格 suite 使用量：0 个 GPU/服务逻辑 evaluator job，0 分钟 GPU active；最多 6 jobs/20 分钟授权未动用。**
本机 torch 2.10.0，CUDA build 13.1，但 `torch.cuda.is_available=False`、无 nvidia-smi 和设备节点。
初始 `unshare -Ur` 成功不代表隔离可用；更完整 mount/proc 探针失败，Landlock ABI syscall 返回 ENOSYS，均未被写成资格。
未用旧 L20D/H20 材料或 synthetic K2 bundle 代签。

## 实际验证与保留位置

自己的环境：`KernelSchema/.worktrees/.tooling/k1-evaluation`，Python 3.13.12；
通过该 venv 内独立 `.pth` **只读引用已存在** `/opt/venv/python3.13.12/lib/python3.13/site-packages` 的 torch，
不下载 CUDA/解释器或修改其它 venv。测试用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 避免全局无关 plugin 缺失。
初次发现 own venv 不继承外层 torch 导致 13 skipped，不计通过；引用已有 torch 后才得到真实逻辑结果。

| 实际命令/检查 | 结果 |
| --- | --- |
| own Python `-B -m pytest -q test_evaluation_logic.py test_evaluation_execution.py test_evaluation_agate.py` | **27 passed，79.13s**；固定原始证据目录如下 |
| Agate transport 安全小修后单独 `test_evaluation_agate.py` | **5 passed，2.27s** |
| 确定性慢候选软目标用例加强后定点测试 | **1 passed，3.15s**；完整采样与 unmet 可同时成立 |
| K2 原隔离 Python 对原 `test_api/test_artifacts/test_contracts/test_template_extract.py` | **59 passed，21.74s** |
| 原 `task/check_materials.py` | 通过；命令末尾 NOT RUN 仍是 B01 该命令自己的非执行声明 |
| 原 golden/baseline、interface/environment 与 K2 共同 fixture 字节 | 保持匹配；没有刷新共同 fixture 指纹 |
| `git diff --check` | 通过 |

固定原始 CPU/transport 目录：
`/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluator-evidence-20261120/`。
最终只读 `GET /v1/env` 的实际 HTTP 401 原始 body 另保留于该目录 `agate-readonly-auth-probe/response.raw`，
摘要 `sha256:fba87d3f282e45423d9c630a94dd7be1b7a9759d31eeb2af963e1aee669f34c5`；这次请求未提交 GPU job。
包括实际 candidate/stage/bundle、frozen-inputs、每调用 output/golden/backing/guard、80 条真实 CPU ns 样本、
独立 controller 原 intent/receipt、ACK 丢失后原终态、重复 absorption 与 fake Agate 原 HTTP/event 字节。
其中错误主体测试故意篡改了测试副本，篡改前真实原件另外保留为 `stdout.before-forgery.raw` 与 `terminal.before-forgery.raw`。

重要原 terminal 摘要：

- 正例 `test_real_child_readonly_full_0/jobs/one/native-terminal.json`：`sha256:86902508a4cac90dbdef8301e67f42015daccbdaf530c72ebd8697557609cf36`。
- 实际 spawn 后 ACK 丢失、原 controller 完成 `test_submit_ack_loss_unknown_f0/jobs/lost/native-terminal.json`：`sha256:4e7cddd916027a52deb8c37275ba51259a447aebffa2648452e726777eef4e44`。

反例覆盖新张量返回、clone 返回、同对象换 storage、只写首行、漏末列、改 x、超差、前/后 guard、状态缓存、域外 shape/alias/device、
实际写 evaluator、staged 字节变更、旧 source receipt、404/cancel/ACK 丢失、迟到和重复收据。
最初真实 full child 暴露绝对纳秒超 safe integer 的失败已修为十进制字符串，未删样本或弱化 JCS。

## 交接边界

K4 可消费本报告开头的 API；准备自己原 RunContext/owner/预约后调用，不能把 CPU unknown 或条件性 Agate engineering 当成 CUDA admission。
K4 仍需完成固定 loop、fidelity、任务语义不足的 unknown 处理、单 owner 总账与结果组合。
未知作业与“已结算但缺某检查”的后续补证是不同状态，接口分别暴露，不能一概用重提或一概禁继续替代。

本批只更改 evaluator 新代码/说明/材料 manifest、evaluation 包、专属 tests/fixtures 及 references/missing.md；
保留 `evaluator/interface.md`、`task/environment.md` 原字节作为历史材料，以免破坏已固定 K2 fixture。
manifest 仍明示自身为 unbound interface_only descriptor，新增 implementation_status/实际组件 hashes/新文档入口；
真正可执行规范 manifest 只能由 freeze_task 输出，不能把材料 JSON 当 Ref 使用。

## 签收前接口定点修正

根据主代理审阅，删除静态 Task/scope 验证器中无条件的 CUDA 未资格结论，将参数改为
`require_hardware_binding`。具体非 pending 硬件文档绑定现可在静态检查中 pass；pending、unknown、
缺失字段仍 unknown。该开关只增加文档绑定检查，不能授予真实执行资格。
真实 CUDA/profile 资格由执行器和 K4 的准入组合负责。

同时固定当前评测范围为全 Task：`evaluation_capabilities/require_evaluation_scope`、
validate_scope 和显式 scope 的 stage 共同拒绝 subset-only，避免部分 scope 后暗中评测全域。
只修改 task.py、相关 scope 测试和接口/报告说明，其他 evaluator/执行器代码继续冻结。

受影响 scope/staging/core-import/capability 检查 **6 passed，3.44s**；原 K2 核心回归
**59 passed，21.33s**；`git diff --check` 通过。未重新执行 GPU/模型或改写既有真实 CPU 原始证据。
