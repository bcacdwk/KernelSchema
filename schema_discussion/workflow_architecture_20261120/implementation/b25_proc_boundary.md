# B25 / explicit inherited-proc stage boundary

2026-11-21。生产开发树 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`，基点固定 **`6db007b3290b59f542316f12fd647028f4901696`**。仅改 KS evaluation、对应测试和执行文档；原 main 仅写本报告，临时证据在 [B25](/tmp/schema-b25-proc-boundary-20261121-gxo18j57)。没有提交/推送/派生代理，没有读取、枚举、修改或运行 AKA_experiment，也没有修改 driver/runtime/预算/Task 算法/Schema/AKA。

**已建立一个有内核依据和本机实测的最小路径：新增显式 `agate-dev-stage-ro-inherited-proc-v1`，保留原受限 `/proc` 及所有 mask/submount，以独立 user namespace 的 ptrace/commoncap 权限和实际父进程 canary 阻断原可写 stage 别名。不是 fresh proc，也不隐藏外部 PID/进程元数据。** 原 fresh profile 继续要求 fresh proc；B20 本机与 B22 L20D 的 mount-denied 失败原件不改，不换 key 重投旧 scope，不把外层 gateway succeeded/cleanup 当权限或 Task pass。

B22原terminal也已独立解压核frame长度/hash：[b22-original-terminal-audit.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/b22-original-terminal-audit.json)。原job仍`dv_80d527ccf029`，terminal SHA仍`9be5354ad6ab0c76be200e06f42d77435e176b86612054ee8ba84eaf11ce49aa`，outer succeeded/cleanup false对应inner exit1、`mount: /proc: permission denied.`，不据此签权限通过。

本阶段新增 **0 GPU/evaluator jobs、0 模型调用**，没有调用或读取真实 owner 账户；不引用旧余额为当前余额。全部服务资格仍待根固定源码后、释放统一 slot，再用同一原 parent 进行一个新冻结 Task/profile 的原六作业套件。没有用这个 CPU 结果签 K1/K4/K6 或 AKA 配对能力。

## 原合同与具体机制

V25 要求候选无 evaluator 文件写权限；C104 要求固定 staging、实际加载 C/evaluator、受信摘要、实际环境和原终态。新 profile 保留完整 stage 不可写和原完整 Task 判据。whole-root readonly 和 fresh/private proc 是旧具体实现的额外边界，不能假装保留；`scope=complete_evaluator_stage`、`whole_root_readonly=false`、`proc.mode=inherited-restricted`、`private_pid_view=false` 明确记录真实承诺。和原实现一样，这不隔离同一 Python 地址空间中的任意恶意 native code。

六个原算子/evaluator 模块与五份原 Task 文件逐字节和基点 Git 比较一致：[criteria-unchanged.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/criteria-unchanged.json)。原72+2、原 x/out/storage/view/guard、调用前 CPU FP64 golden、逐元素容差、完整 forward 与四形状各80 AB/BA批没有变动。`task.py` 仅在新冻结 evaluator 的 adapter 列表加入新 profile，旧发布对象不原位改写。

已有 pinned Python/unshare/mount/setpriv 继续由 probe 前配置固定 hash，并在实际 launcher/worker 检查。`unshare --user --map-root-user --mount --pid --net --uts --ipc --fork --kill-child=SIGKILL --propagation unchanged` 后，六 namespace 独立性 guard 先于一切 mount。然后只在本 job 独立 mount namespace 中设置 private propagation、bind/remount 完整 stage 为只读。没有尝试 fresh proc mount，没有卸载/覆盖原 proc mask，没有宿主 remount、chmod、sysctl、服务/镜像更改或包安装。

完整 stage 仍拒 symlink/hardlink/特殊文件、子挂载、同 backing 的另一可写 mount 别名；worker 在 scratch cwd、只持标准IO，cache/HOME/TMPDIR 指向 scratch。cap Inh/Prm/Eff/Bnd/Amb 均0、NNP=1。每个实际 stage 文件及新建文件都必须 EROFS；scratch 必须真可写。模块从已核 hash 的实际 byte buffer 编译；执行后 worker 和 outer launcher 都复核 stage 路径/原字节。

## 内核论证与固定官方出处

Linux proc root/cwd/fd magic link 的 readlink 和 follow 都使用 `PTRACE_MODE_READ_FSCREDS`。相同 UID/GID 只满足前层检查；始终参与的 commoncap 还要求“相同 user namespace 且 caller effective capabilities 覆盖 target permitted capabilities”，或“caller 在 target user namespace 具有 CAP_SYS_PTRACE”。子 user namespace 的 capabilities 不授予父/祖先 namespace 权限。

- outer Python launcher 保留原 user/mount namespace，持有真实原 stage 目录 FD 和 O_RDWR 文件 FD，保持到 child 完成；这些 FD 不传给 child。worker user namespace 不同，因而不能通过 outer 的 root/cwd/fd 恢复原可写视图。实际针对已知父进程 canary 验证 EACCES/EPERM，ENOENT 不算权限证明。
- `unshare --fork` 的等待父进程处在 child 的新 user/mount/net/uts/ipc namespace，仍属于旧 PID namespace。实测并验证其直接父链、这六项拓扑、非零 permitted capabilities、scratch cwd，以及仅标准IO和至多一个可验证的内核self-pidfd（见下）。child cap 清零后，commoncap 的 capability-superset 条件不成立；该 helper 的 proc 别名也被拒绝。其共享 mount namespace 本身已经看到 stage RO。所有同步 mount helper 已在 candidate import 前退出。
- 自身 proc magic links 没有跨进程限制，但 root/cwd/只在bind后新开的目录FD都解析到受保护 mount；三个真实写开 canary 均要求 EROFS。没有把原 cwd、祖先/文件FD或其他 mount alias 传入 worker。
- candidate 后续若再建 user/mount namespace 以取得子 namespace capabilities，`copy_mnt_ns` 的 `lock_mnt_tree` 会锁定继承 RO 和覆盖 mount；`can_change_locked_flags` 拒绝转rw，不能卸载锁定子挂载。`__do_loopback` 同样拒绝从有 locked child 的父目录创建非递归 bind/open_tree 别名；递归克隆保留 protected mount。普通跨 bind mount 新 hardlink 也不能把文件导出为可写别名。

官方 Linux v6.6 来源原字节与 Git blob/sha 索引：[kernel-api-index.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/sources/kernel-api-index.json)。保存了以下足够核查的文件，没有网络漫游：

| 原件 | 固定 Git blob | 必要位置 |
| --- | --- | --- |
| security/commoncap.c | `bc05211041979c120831742d25f2e6133435166c` | cap_capable:66–98；cap_ptrace_access_check:134–157 |
| kernel/ptrace.c | `443057bee87cb7469990f3ad1d6abe09a42c22f3` | __ptrace_may_access:287–359，凭据/目标dumpable后仍调用security_ptrace_access_check |
| fs/proc/base.c | `ffd54617c35478e92a9f6bef67013e16e6cd3183` | proc_fd_access_allowed:673–688；proc_pid_get_link/readlink:1741–1812 |
| fs/namespace.c | `e157efc54023a051faa7dcf0b7d91813dbec7134` | lock_mnt_tree:2150–2177；__do_loopback:2575–2598；open_detached_copy:2648–2685；can_change_locked_flags:2744–2765；copy_mnt_ns:3744–3783 |

另保存官方 man7 proc_pid_root(5)、proc_pid_fd(5)、ptrace(2)，[man-index.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/sources/man-index.json)固定 URL/SHA/长度，ptrace access-mode算法第3–5步与内核代码一致。最初固定 raw GitHub URL 返回429，固定 kernel.googlesource URL timeout；改用同一官方仓库固定v6.6 contents GET 获得原件，没有执行下载代码或安装任何材料。

另外有一个与L20D现有工具版本直接相关的细节：本机unshare为2.32.1，L20D已观测为2.39.3。固定官方 [unshare-v2.39.3.c](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/sources/unshare-v2.39.3.c) 的942–1032行，在支持pidfd的构建上会于fork前打开自身pidfd，等待父进程持有它，child在exec前关闭；Git blob `29fad711421bb6f0a9aadf2da24a29c57cd76b09`、SHA与URL见 [util-linux-index.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/sources/util-linux-index.json)。因此最终实现只额外接受至多一个 `anon_inode:[pidfd]`，还须从这个已知helper的fdinfo证明Pid正是helper自己，并纳入cap-drop后的拒绝canary及资格校验。不是允许任意额外FD；普通文件/目录或指向其他进程的pidfd都拒绝。本机受控真实父子测试验证了self-pidfd可识别、指向原父进程的pidfd被拒；合成六作业通道覆盖2.39式self-pidfd记录贯通，不能冒充已经实测远端2.39分支。

## 实际 CPU 证据

最初独立 [proc-canary-result.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/proc-canary-result.json)：trusted outer 真实 cwd 在原stage，持原目录FD及O_RDWR文件FD；child完成独立namespace、private propagation、stage ro、cap清零/NNP后，已知outer的root/cwd/目录FD/文件FD写别名均EACCES(13)。同user/mnt的unshare parent仅标准FD，cap drop后root/cwd亦EACCES。自己的proc/root与直接stage写开EROFS(30)，scratch可写，parent原字节未变。没有扫全/proc或读取其他用户进程。

本机内核实际是 `5.10.112-005.ali5000.alios7.x86_64`，uid1000，[local-environment.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/local-environment.json)。B19所见L20D为6.6.102；固定上游v6.6规则与本机实测共同支持实现，但不能代替L20D自己的真实权限/CUDA资格。

生产 helper 原真实运行 [local-production-result.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/local-production-result.json) exit0：父进程九项别名拒绝、自己三条proc路径EROFS、完整stage/新建文件EROFS、六namespace、五cap零/NNP、proc及原mask/submount未改、scratch可写均由真实代码测得。该 local probe worker 仅调用生产 permission verifier，没有伪造 readonly字段，也没有导入CUDA，因此不能声称完整GPU worker/Task已运行。

最终字节对应 [final-local-production-result.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/final-local-production-result.json) 仍exit0。outer和helper的cwd现直接构造写入stage的proc路径并要求EACCES/EPERM，连同原root/FD写canary及self三路径EROFS一起通过。旧fresh分支最终原输出 [final-fresh-result.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/final-fresh-result.json) 仍mount /proc permission denied、inner exit1；这些CPU重测不涉及原B22的job/key/owner。

[local-nested-bind-result.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/local-nested-bind-result.json)在实际profile内完成第二层独立user/mnt，再通过直接libc syscall保留新namespace中真实恢复的 CapEff=`000001ffffffffff`：stage remount-rw得EPERM(1)，卸载stage和对stage父目录非递归bind都EINVAL(22)，原outer文件FD仍EACCES(13)，stage仍EROFS(30)；新hardlink得EXDEV(18)，chmod得EROFS(30)。每个mount syscall前均已确认相应独立user/mnt，没有宿主mount操作。

相同nested用例以最终helper字节复测，原输出另存 [final-nested-result.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/final-nested-result.json)；早期原件未覆盖或重写。

实测外部 `/proc/<parent>/fd` **编号列表仍可见**，所以明确不承诺PID/metadata隐藏。这里的边界是magic link解引用权限和保护mount，而非全proc保密。

## 接线、验证与后续

新 profile 贯穿 selection、prepare、实际request/command、remote dispatch、worker实际profile、frame validator、read_qualification、admission、原key/locator恢复；FA4仍bwrap-only。旧fresh资格和新inherited资格相互不能重标；proc mode/parent identity/helper namespace/caps/fds/实际拒绝/自身EROFS缺项均不能产生合格报告。原完整loaded source/evaluator、metadata、frozen inputs、截断frame、真实terminal/cleanup条件保留。没有短路径跳过六作业，也没有owner/费用旁路。

初轮新旧profile定点测试 **45 passed、1 skipped（38.67秒）**。随后六个相关测试文件回归 **119 passed、1 skipped（243.31秒）**，[cpu-regression.log](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/cpu-regression.log)，覆盖channel、Agate、execution、logic与FA4。最后的self-pidfd与直接cwd canary细化后，受影响的新旧profile定点回归 **50 passed、1 skipped（35.80秒）**，[final-stage-ro-tests.log](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/final-stage-ro-tests.log)，包括合成原六作业/恢复/准入、缺失或伪造proc证据、旧stage FD泄漏拒绝、真实self-pidfd身份和nested逃逸。所有唯一skip均为旧fresh-proc本机挂载拒绝，新inherited本机真实测试没有skip。`git diff --check`通过；没有安装pytest，使用既有k1-evaluation解释器并关闭环境自带plugin自动加载。

最终10个修改/新增文件的长度/SHA与基点记录于 [cpu-source-freeze.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/cpu-source-freeze.json)，机器交接为 [b25-final-handoff.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/b25-final-handoff.json)。根提交后应以新Git SHA绑定真实新材料；本代理没有提交或用未冻结变化运行真实服务。

后续配置 [remote-stage-ro-inherited-proc.json](/tmp/schema-b25-proc-boundary-20261121-gxo18j57/remote-stage-ro-inherited-proc.json) 只在B20原L20D实测四工具配置中显式选择新channel，没有猜测工具身份。根提交本实现后，新的独立GPU阶段直接运行原六job suite即可；首probe必须在L20D实际产生全部新permission与CUDA/input证据，失败仍停原suite。无需为本方案另发能力诊断job。当前不声称L20D已经通过这一新profile。
