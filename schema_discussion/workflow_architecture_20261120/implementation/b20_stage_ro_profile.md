# B20 / fixed evaluator stage-ro profile

2026-11-21。唯一生产开发树为 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`，起点 `77ad741ca5b088d24f4cd23018eb0efcd491f15f`。本代理不提交、推送或派生代理；不读、枚举、改或运行 AKA_experiment，不改 AKA、runtime/claude.py、预算、原 Task 判据或原 Schema。原 main 仅写本报告。

本阶段实现独立的 **`agate-dev-stage-ro-v1`**，使用 B19 已观测的现有 util-linux 工具保护完整 evaluator stage。它明确不承诺 whole-root readonly，不借用旧 bwrap 的资格。旧 `agate-dev-bwrap-v1` 分支保留；FA4 仍只接受原 bwrap profile。

证据根：`/tmp/schema-b20-stage-ro-20261121-c_irxa_5`。当前是 CPU 接线与源码冻结阶段，**新增真实模型调用 0、GPU/evaluator jobs 0**。尚未触发原 parent 的任何新收费操作，真实链没有新 owner；测试临时合成账户没有服务调用。后续 L20D 原六作业只能在根固定源码版本并释放统一 slot 后，使用同一原 parent 执行。

## 执行边界

固定 `_stage_ro.py` helper 属于 evaluator 闭包；新增约 200 行固定工具调用与检查，没有通用 sandbox DSL、镜像/工具安装或下载逻辑。`_remote.py` 只按显式 profile 选择 helper。工具四元组为现有 Python、unshare、mount、setpriv；路径和真实 binary SHA-256 在 probe 前固定，每次 launcher/隔离 worker 复核，包括 `/proc/self/exe` 的实际 Python 字节。

launcher 启动时 cwd 已切到外部 scratch，关闭非标准 IO FD。`unshare --user --map-root-user --mount --pid --net --uts --ipc --fork --kill-child=SIGKILL --propagation unchanged` 后，必须先证明六类 namespace 全部独立，才允许执行 mount。随后将传播设为 private，挂载新 PID namespace 的 `/proc`，只对本 job 完整 stage 自身 bind/remount-ro。拒绝 stage symlink/hardlink/特殊文件、外部子挂载和通过另一 mount 暴露 backing 的别名。没有宿主 remount/chmod。

setpriv 清空 inheritable/permitted/effective/bounding/ambient capabilities，设 `NoNewPrivs=1`；worker 在 torch/evaluator/candidate 导入前核查实际状态、PID 1 与新 `/proc`、独立 namespaces、private mounts、scratch cwd 和无遗留非标准 FD。完整 stage 的全部文件（含实际 runner、stage/remote/execution 元数据、frozen inputs）逐个 O_WRONLY 必须得到 EROFS(30)，创建 stage 文件也必须失败；另外实际写 scratch 作对照。HOME/TMPDIR/Triton/XDG/CUDA caches 指向 scratch，保留实际设备挂载供 CUDA 验证。

没有另外留下可写 evaluator 加载副本；真正加载的唯一 stage 和 metadata 均受保护，模块从已验证的 byte buffer 编译，候选后还核 stage 原路径/字节。父进程超时杀整个进程组，unshare 父退出会杀 namespace PID 1。gateway 原完整终态与 cleanup 检查仍必须通过。这不是对任意恶意 native code 修改同一 Python 地址空间的安全证明；既有 CPU/bwrap 边界同样不作该承诺。

## 贯穿原链

新 profile 贯穿 channel selection、实际 prepare/request/command、remote settings、worker frame、原 terminal 解释、qualification/reinterpretation、后续 admission 和原 locator/key 恢复。证据使用 `scope=complete_evaluator_stage`、`whole_root_readonly=false`，没有冒填旧 readonly-root 属性。缺 namespace、proc/cap/FD/probe/metadata/工具哈希或实际 loaded 字节不符均无报告/不准入；truncated native output 不可能签 pass。

沿用原 `qualify()` 的六个 phases：同 profile 的 CUDA/输入 probe，完整 baseline，以及 missing_out、changed_x、tolerance、storage 四负例。新 profile 的 probe 自身也进入完整隔离边界。原稳定 key、unknown fence、只查原 job、原费用与 terminal/cleanup 逻辑没有旁路或第二账户。资格与旧 bwrap 不能互换，工具变化需另作新主体资格。

六个算子模块、输入/比较器、72+2 调用、原 x/out/storage/guard 规则、调用前 CPU FP64 golden、完整 forward 及四形状原 80 AB/BA batch 不改。`criteria-unchanged.json` 对六个模块与基点 Git 原字节逐一核对。新 helper/worker 字节改变 evaluator 身份，后续必须新冻结 Evaluator/Task/Q；旧 B16 Task/Q/Schema 和已发布材料保持不变。

## 当前本地证据与边界

本机真实六 namespace unshare 入口成功，但独立 namespace 中建立新 `/proc` 被权限拒绝（mount exit 32），stage bind 尚未执行。尝试只读 proc 同样被拒；没有放宽新 proc 要求或 fallback。生产 helper 原始 stderr 与结果保存在 `local-namespace-result.json`，父 stage 前后原字节一致。真实 local stage-ro 成功测试因此明确 skip，不能把本机 namespace 入口当只读成功。

B19 在 L20D 证明了独立 namespaces 和临时 stage 型目录 ro，但未建立新 proc。因此，本机失败既不证明 L20D 必败，也不能证明 L20D 新 profile 可行。实际原六作业首个 probe 必须补足该项；若失败，原 suite 停止并保留原费，不追加伪资格或偷偷换机制。

第一次 pytest 启动遇到环境自带 `platformdirs.pytest_plugin` 缺失；使用既有解释器设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 后运行，没有安装/改环境。新 profile 定点测试首轮 17 passed、1 skipped，覆盖六作业原 key 恢复、只读属性虚报/缺项、工具/metadata/loaded 字节、hardlink/symlink/mount alias、缺工具及 namespace guard。

最终定点回归 **90 passed、1 skipped（198.85 秒）**：`test_stage_ro_profile.py`、`test_agate_channel.py`、`test_evaluation_agate.py`、`test_evaluation_execution.py`、`test_evaluation_logic.py`、`test_fa4_channel.py`。唯一 skip 是上面说明的本机 fresh proc mount 权限缺失；没有 CUDA skip 被计成 pass。另补强“修改上传字节且同步重算 request digest”拒绝、stage-ro 丢失原 acknowledgment 只结算不重投、实际工具 hash 不符先拒绝，相关两项 **2 passed（3.62 秒）**。后续 channel admission/prepare/submit/settle/repeated submit 也已由新 profile 的合成传输实测覆盖。

`git diff --check` 通过。没有全库重跑。源码以基点 HEAD 和逐文件哈希冻结，索引 `cpu-source-freeze.json`，根提交后再把真实新 Git SHA 绑定到后续输入。当前不声称任何 L20D 原六作业或业务链通过。

L20D 固定配置在 `remote-stage-ro.json`，工具来源索引在 `tool-provenance.json`。它仅复用 B16/B19 已实测二进制身份，不把先前 namespace 诊断充当新 profile、CUDA、完整 Task 或后续 KS/AKA 业务资格。
