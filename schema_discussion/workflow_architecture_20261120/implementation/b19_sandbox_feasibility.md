# B19 / L20D sandbox feasibility

2026-11-21。只读可行性研究；KS 固定 `7d5fb60cf1d6f9ef7cf63dff757250bb4f614ab1`，AKA 固定 `89fb61b696a93a255f44668658f79bd8bd0e6ce3`。本代理仅写本报告与 `/tmp/schema-b19-sandbox-20261121-_s6bxx0v`；不改源码、安装、服务、sysctl、AGENTS 或旧证据，不提交/推送，不派生代理，不访问/枚举/运行 AKA_experiment。使用 [atrex-gpu-gateway skill](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/SKILL.md)。

## 决策

**推荐以现有 `unshare/mount/setpriv` 实现一个固定、有限的 `agate-dev-stage-ro-v1`：保护完整 evaluator staging，保留全部 Task 判据与权限要求；不承诺整个 worker 根文件系统只读。B19实际诊断已证明所需namespace、临时bind/remount-ro、cap清零/NNP和拒写/可写对照可运行，无需先安装bwrap或改镜像。**

这不是沿用旧profile删检查。新profile必须明确声明边界、绑定受信launcher/工具hash，并用原六job套件建立自己的资格。`bwrap`仅是旧profile的实现选择；若仍要求旧profile的whole-root readonly，则优先固定job-local bwrap，而不是手写递归root挂载平台。sealed memfd可作无需namespace的窄备选；普通trusted-process不满足原权限项，不推荐。

L20D 已有真正成功的 dev 诊断 `dv_37ffeb8cd2f2`，并完成 cleanup。实测 `/venv/bin/python` → `/usr/bin/python3.12`、Python3.12.3、`"NVIDIA L20D        "`（8个尾空格）、SM10.3、Torch2.9.0+cu130 / Triton3.7.1 / CUDA13.0；UID/GID=0，`which(bwrap)=null`。B16当时namespace/readonly尚未执行；本轮新增实测见末节。不能说 dev 不可用、AK 无权，也不能说所有路径中均不存在 bwrap。[B16报告](/home/shaohanyong.shy/KernelSchema/schema_discussion/workflow_architecture_20261120/implementation/b16_l20d_real_validation.md)及[交接](/tmp/schema-b16-l20d-20261121-n5tg6rp7/b16-final-handoff.json)保留原件。

本报告把下列三层分开。第1层可在不同 launcher 中保持；第2层不能因算子结果正确而删除；第3层不能由本任务推定管理员已经作出更强承诺。

| 层次 | 原合同/实现规定 | 可以改变的范围 |
| --- | --- | --- |
| 算子与测量判据 | 72基本调用+2次同对象复用，原 x/out storage/view、完整输出和外围 guard，CPU FP64 调用前 golden、原逐元素容差；四形状原80批 AB/BA完整 forward timing | launcher 可换；不改变 ABI、输入、检查器、样本或时序，不把 harness 藏进 candidate，不让 wrapper 补写 out |
| 测量可信性与权限 | V:25 明示“候选无 evaluator 文件写权限”；C:104 要固定 staging、核实际装入 C/evaluator、受信摘要/环境/终态；V01/V06据此解释实际结果 | 可用等价机制实现，bwrap 不是规范中的必需程序名；但只校验 hash 或候选自报 pass 不等价 |
| 当前具体执行与服务边界 | `agate-dev-bwrap-v1`：pinned executable，`--unshare-all`，根只读，独立编译 scratch、`/dev`、fresh process、cap drop；dev 要recycle/never replay/完整终态及cleanup | 替换机制必须显式新 profile/新主体与资格；不能保留旧 profile 名字而跳过检查。网关文档另外要求部署方建立网络/执行隔离，不能从一次 dev 成功推断所有部署属性 |

依据：[完整Task/V25](/home/shaohanyong.shy/KernelSchema/schema_discussion/workflow_architecture_20261120/planning/r5_validation_delivery_plan.md:14)、[C104–106](/home/shaohanyong.shy/KernelSchema/schema_discussion/workflow_architecture_20261120/planning/r5_contract_execution_review.md:104)、[EXECUTION](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/examples/workflows/row_softmax/evaluator/EXECUTION.md:142)、[operations](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/reference/operations.md:8)。原 bwrap 也没有把 candidate 与 evaluator 分离到两个地址空间；这不是对任意恶意 native code 的完整安全证明。原CPU sandbox明确说明这一局限且不支持CUDA；不能为GPU直接复用它。[原限制](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/evaluation/_sandbox.py:1)

## 少量有界路径

| 路径 | 可行前提与现有证据 | 结论 |
| --- | --- | --- |
| 现有 `unshare` / `mount` / `setpriv` | B19已实测六类独立namespace、临时stage型目录只读、scratch可写、cap清零+NNP。固定stage可作为单个无额外子挂载的保护树；实际worker/CUDA/进程清理仍待资格 | **首选有限stage-ro profile**：保持V25的evaluator保护，不承诺旧profile的整个root只读。新/proc、cwd/FD/原可写别名处理与完整stage范围必须闭合；不扩成通用root sandbox |
| job-local固定bwrap | 官方dev `files`支持相对路径→文本；可明确传base64文本，再由trusted launcher只在本job临时目录解码/核hash/设可执行位，不改镜像/系统，无pip/apt。实际二进制、架构/动态依赖、来源与许可、请求体限制和namespace仍须证明 | 实测namespace支持给此路径正面条件；若确需原whole-root边界，这是最接近原bwrap配置的路径。但当前无已验便携binary，尚需artifact/依赖运输；不是首选新增依赖 |
| sealed evaluator memfd | Linux内核不可逆文件封印可禁止内容修改，无须namespace、安装或新binary；本轮只有官方API静态证据，未测worker支持/完整loader | 有条件满足evaluator内容不可写的实质，需新profile与真实seal/拒写证据；不能沿用旧write-open-denied字段。不必因工具缺失先降到普通trusted-process |
| 明示 `trusted-process` | 固定受信harness/fresh process/实际loaded字节与完整frame仍可运行全部算子检查；若无OS只读边界，候选仍有evaluator写权限，且没有原root/namespace承诺 | 可产有明确局限的真实数值/性能诊断；不能签当前K1完整资格或由它导出的K4/K6等门。必须改变执行合同/验收口径才能把它作为新的支持能力，不能默默降级 |

不把 ordinary eval / custom SOL 列为已可行捷径。B14已读公开接口，尚未找到满足原72+2、原storage/副作用、golden64、原80样本、受信hook/readonly及完整报告的等价路径；本轮没有对其另投收费job。[B14结论](/home/shaohanyong.shy/KernelSchema/schema_discussion/workflow_architecture_20261120/implementation/b14_agate_real_validation.md)

## job-local bwrap 的具体可行边界

只读官方资料已固定到 `containers/bubblewrap@362936aea0d5c6e70f7af75d2d44b5a914cb4ba2`（tag `v0.13.1`），源API原件、blob/hash见 [来源索引](/tmp/schema-b19-sandbox-20261121-_s6bxx0v/sources/bubblewrap-index.json)。官方latest资产列表只有 `bubblewrap-0.13.1.tar.xz`，129472字节，发布digest `0e85c8702248b8dd64144f594147553650cdd0e4c951690345faace79c217c4d`；这是源码，未下载/编译/安装。没有证明“官方已给一个可直接复制的静态binary”。

- README:15–22说明以user namespace构造sandbox，旧setuid模式已移除；:44–60明确bwrap不是自带完整安全策略，实际边界由调用方参数决定。
- `bubblewrap.c`开头SPDX为`LGPL-2.0-or-later`；仓库COPYING文本为LGPL2.1，二者原字节均保留。分发二进制应带原授权通知、对应源/构建来源及其链接依赖的许可材料；不能写成MIT或忽略静态依赖义务。
- `meson.build:68`必需libcap，libselinux可选。普通发行版动态binary可能依赖worker没有的loader/库；要提供实际目标架构、ELF/NEEDED、各依赖digest和运行证据。静态artifact可以另行构建，但不能仅凭“便携”命名声称没有依赖。
- 当前源码:3057–3124以`raw_clone`创建所选namespace；`unshare`探针与其不是同一个syscall。即使`unshare`返回EPERM，也不能单凭这点证明所有bwrap执行必败；更不能据sysctl允许就宣称clone/mount被服务安全策略允许。
- 公开`/docs/api/dev`把`files`定义为“工作区相对文件路径到文本内容的映射”；[jobs reference:73](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/reference/jobs-and-troubleshooting.md:73)明确dev使用`--file`，不支持其OSS附件选项。故不能直接把任意二进制塞入UTF-8字段。显式base64 staging是普通文本载荷，但必须遵守实际dev总量/单文件策略；已见L20D capabilities的`max_source_bytes=131072`不能擅自解释成无限dev文件额度。
- 不需要凭空发起新的用户审批：原已授权custom harness/dev与job-local临时依赖不等于系统安装；若实施选择改变了源码范围或合同，由根明确分配新任务。真正需要管理员的是现有内核/安全策略不允许所需namespace、可执行挂载或隔离时的部署侧调整，或提供已具能力的获准L20D worker。安装一个bwrap不能自动修复这些能力。

## sealed memfd 的静态比较

官方Linux man-pages说明[原件索引](/tmp/schema-b19-sandbox-20261121-_s6bxx0v/sources/memfd-index.json)：`memfd_create(MFD_ALLOW_SEALING)`可创建内核匿名文件；填充后加入`F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL`，封印不可移除，写入/改变长度/新建共享可写mapping被内核拒绝，已有O_RDWR fd也不能写。`F_SEAL_WRITE`在存在旧共享可写mapping时会EBUSY；单用`F_SEAL_FUTURE_WRITE`不足以保护已有mapping。它是文件内容保护，不只是Python bytes或chmod0444约定。

要满足本合同，所有实际权威evaluator代码/配置/输入必须从已核hash且已确认完整seals的fd装入；不能只封印一份副本，同时让原可写staging继续作为worker、module `__file__`、再次import或检查证据来源；候选入场前须清除不再需要的原可写副本/别名，不能只换一个Python变量引用。需处理fd继承/关闭、实际loaded主体、最后封印检查，以及可写别名和fd号码替换；从fd编译已验证字节不再按可变pathname读入。候选与evaluator同地址空间的限制仍与原实现一样，不宣称防御任意恶意native code。

旧`prove_readonly`只尝试O_WRONLY打开；memfd有seal时打开仍可能成功，写操作才EPERM。因此需要显式`sealed-evaluator` profile及真实`F_GET_SEALS`、拒写/拒mmap和实际hash证据，不能谎报旧`write_open_denied=true`。它可以保持V25“不允许修改evaluator文件”的实质保护，但放弃whole-root/network/PID隔离。新的有限qualification可按它真正保证的属性解释；原bwrap profile和旧K1 gate的字面声明不能直接沿用。本轮不实现或测试memfd，不把它列为已建立资格。现有namespace已实测可用，故当前无需先开发这条备选。

## 最小实现范围与验证

保持六个Task模块、比较器、输入/timing recipe原字节。建议新任务仅实现固定`stage-ro` profile，代码限`evaluation/{agate,channel,_remote,_worker}.py`及现有对应测试/EXECUTION/protocol说明；允许一个很小的固定launcher helper，不引入通用隔离DSL、namespace库、下载器、包安装器或镜像管理。

最低闭合条件：

1. 原K1 request所含完整stage（源/六模块/全部worker、manifest、execution、frozen inputs）形成唯一保护树，拒symlink/意外hardlink/外部子挂载与可写别名；实际文件hash和profile/tool hash绑定原intent。不要仅保护六个`.py`却留下可变执行配置。
2. 固定`unshare --user --map-root-user --mount --pid --net --uts --ipc --fork --kill-child`进入子namespace，验证身份后只把本job的stage bind/remount为ro；明确不把整个`/`remount成ro。用新PID namespace对应的新`/proc`阻止通过外部进程root/cwd/fd返回旧mount视图；不能把本诊断未挂新proc的检查脚本原封不动当生产launcher。
3. worker在外部compile scratch作为cwd，所有cache/HOME/TMPDIR指向该scratch；exec前关闭原可写stage目录/文件FD，不能留一个早于ro bind的cwd或fd绕过路径。完整cap set清零、NNP=1；检查工具hash/实际状态，父子生命周期/timeout与recycle必须可核。没有namespace/拒写证据立即fail closed。
4. 已隔离worker在candidate import前，对实际stage执行原拒写probe并记录真实profile；完整Task、loaded字节、inputs/environment、frame、终态/cleanup仍由同原链验证。CUDA在该profile内必须实际运行，诊断成功不能代替。

以上是单一固定harness的窄封装，不承担任意路径/任意rootfs/任意容器平台。若这些边界无法用短小且可审计的现有工具调用完成，停在缺项并回到固定job-local bwrap；不要持续扩展自制sandbox。若选bwrap，使用既有Publication/Ref/hash固定唯一工具artifact并验证decoded字节/依赖，原bwrap参数保持；不得现装包或现场下载。

必须同步原request校验、frame消费和qualification主体绑定：当前[agate.prepare/validate](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/evaluation/agate.py:98)、[interpret_terminal](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/evaluation/agate.py:289)、[channel.read_qualification](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/evaluation/channel.py:98)以及[_worker](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/kernelschema/evaluation/_worker.py:56)都明确绑定旧profile/bwrap与permission probes；只改launcher无法合法通过。工具来源/字节/许可须进入选定闭包，不能在模型资料内泄露私有evaluator/凭据。

测试只覆盖这次改动的真实风险：工具/上传/hash/profile任一变化被拒，缺工具/不可执行/namespace失败不fallback，stage可写别名/cwd或FD逃逸不被当作readonly通过、实际拒写/可写scratch、原frame损坏/截断不能pass，旧qualification不能被新profile接受，断连只查原job；随后同一原parent真实跑原六job probe+完整baseline+四负例，在实际L20D核CUDA/readonly/全部判据、80样本、loaded hash、terminal+cleanup。诊断不替代这六个job。新remote/worker属于evaluator闭包，所以即使算子规范不变也须新冻结Evaluator/Task/Q身份，保留旧Task与费用，不原位改旧已发资料。新stage-ro或sealed profile在满足原V权限实质及全部资格之后，可签明确标注该profile的有限Task能力；这不是原whole-root bwrap能力。若根要求原profile所有额外承诺也不变，则它们不是等价替换，选bwrap。

若只实施trusted-process，则`permissions`必须记录not_enforced/unknown等实际状态，不能造`write_open_denied=true`；选用新profile并保持原profile继续fail closed。该结果可附真实完整算子raw evidence，但原[V01/V02/V06及门依赖](/home/shaohanyong.shy/KernelSchema/schema_discussion/workflow_architecture_20261120/planning/r5_validation_delivery_plan.md:136)不因此全部通过：K1.real不签，依赖它的K4.real/K6.demo/release与后续K8.real不签；AKA A2.real/A5.real/A6.real/A7.pair更不能由KS结果代签。数值硬项pass、性能goal与完整支持资格须分别报告。若另设只保护evaluator文件的非root DAC profile，仍要证明不可修改/不可重新赋权/实际CUDA和清理，并明确放弃原只读root/namespace承诺；本轮未测试，也不把它当已建立等价路径。

## 唯一实际能力诊断与费用

根明确分配串行收费slot后，使用既有SDK0.14.37解释器`/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python`，从固定KS Git仅导出`kernelschema/`到本/tmp私有snapshot，不安装/读取在改runtime代码。来源见[source-snapshot](/tmp/schema-b19-sandbox-20261121-_s6bxx0v/source-snapshot.json)。认证仅官方SDK读取已有私有配置，不打印/复制凭据。

一次HTTP202 `POST /v1/jobs/dev`，`recycle=true/replay_policy=never/dev_intent=inspect`，worker60s/request300000ms、最多1job。原operation/key **`b19_l20d_namespace_65ae3e2ce8fc4368b1ca5018ec295917`**，原job **`dv_dca077c701c8`**。随后仅GET该job；succeeded、execution_phase=terminal、cleanup_pending=false、execution_quarantined=false、exit0、output4180 bytes、未截断。没有第二POST/换路由/重复执行。首次本地owner脚本在任何request或account注册前因`.runtime`父目录尚未创建而退出；创建本私有目录后才发出唯一POST，无额外服务job。

[诊断摘要](/tmp/schema-b19-sandbox-20261121-_s6bxx0v/diagnostic-observed.json)来自[原terminal](/tmp/schema-b19-sandbox-20261121-_s6bxx0v/.runtime/diagnostic/terminal.raw)，SHA256 **`23fee0d607fdab7df5a220a0d11630db450c77c9e7dab4e68f3942e793ac6852`**；原请求与脚本hash另见[preflight](/tmp/schema-b19-sandbox-20261121-_s6bxx0v/diagnostic-preflight.json)及最终handoff。

| 实测 | 结果 |
| --- | --- |
| kernel / machine | `6.6.102-5.500.al8.x86_64` / `x86_64` |
| 既有工具 | `/usr/bin/unshare`、`/usr/bin/mount`、`/usr/bin/setpriv`均util-linux2.39.3，绝对路径/二进制hash已记录；bwrap/busybox未被which解析 |
| 内核允许值 | `unprivileged_userns_clone=1`、`max_user_namespaces=16509075`；AppArmor该sysctl路径ENOENT，不推断完整LSM策略 |
| namespace | 独立user/mount/pid/net/uts/ipc六项inode均与父不同 |
| 临时只读目录 | 确认namespace独立之后`mount --bind`和`remount,bind,ro`均exit0 |
| 降权与对照 | setpriv后`CapEff=CapBnd=0`、`NoNewPrivs=1`；protected `O_WRONLY`得EROFS(30)，另一个scratch可写；原control字节未变 |
| 未测 | whole-root/嵌套挂载、完整stage逃逸边界、该namespace内CUDA、完整Task/80样本、生产launcher cleanup、sealed memfd和真实bwrap |

没有remount宿主根、在宿主namespace修改权限、安装、sysctl/服务改动；所有mount仅在已证明独立的子namespace、针对本job的临时scratch。诊断脚本只是能力证据，不作为新生产sandbox交付。原六job资格未启动，K1.real仍未建立。

原parent只新增本次1job/0model，request active **85222ms**，GPU原生时长和货币费用没有由收据提供，保持unknown。四个operation各一条原receipt、全部terminated，active_run/active_since=null、fences=[]，已将收费slot交回根。[账户快照](/tmp/schema-b19-sandbox-20261121-_s6bxx0v/diagnostic-account-check.json)时间 **2026-11-21 03:45:00 UTC**：累计model1/job2/active **207911ms**，剩余model23/jobs14/active10592089ms。请求向下取整相加与parent纳秒累计后取整相差1ms，以parent为准；后续消费属于后续快照。

全会话运行时模型仍累计8次；B17已知CLI费用0.758005USD且原因`budget_exhausted`，不是timeout。原H20拒绝、B16 L20D成功、B17失败、3个旧120秒timeout scope及unknown费用历史不改写。本报告不把新模型调用0写成全会话使用量0。
