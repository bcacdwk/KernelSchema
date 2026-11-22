# 巡检 66

采样与完成时间为 2026-11-20 06:39:37 北京时间（UTC+8）；`clock.committed=true`，下次到期为 07:09:37。

采样时 GDN 已进入 P5，两个非 root 分支并行；2 项 P4、9 项 P3。MLA NVFP4、masked GEMM、FMHA BF16、分页索引 Sparse 四项停于 P3 短交接回执校验；其余 8 项运行，共 9 个在途调用。

两个新的 xhigh 子代理分组核查并在原 run、原预算和正常门禁内处理。恢复前逐项核验真实回执、已用次数、会话、终止确认与控制屏障，禁止直接改状态/旧回执、改阈值或新建 run。

初步核验：MLA P3 为 4/32 次，handoff 3241 字符/5 行，前置实现入口已修复，本 scope 首次暴露长度问题。FMHA P3 为 7/32 次，handoff 已从 2270 缩到 1235 字符/4 行，仅超限 35 字符；Supervisor 根据实际收敛结果同意最多一次原会话续接，不追加额度或循环恢复。

其余两个任务在上次入口修复后暴露原已记录的后置 handoff 问题；子代理须据新结果分别判断。详细恢复证据见 [group_a](../recovery/p3_handoffs_check66/group_a/) 和 [group_b](../recovery/p3_handoffs_check66/group_b/)。

group_b 证实 masked GEMM 已新增绑定 implementation 证据、Sparse 已新增绑定 host 证据；新 handoff 分别为 2815 字符/6 行、2326 字符/1 行，当前 P3 均为 7/32 次。这是入口修复后的新长度诊断，不是旧回执复用。

四项均已各成功提交一次普通恢复：MLA P3 为 4→5/32，其余三项为 7→8/32；新调用均通过原生 init 确认沿用原 reviewer 会话并收到真实诊断。FMHA 首次控制命令返回 `busy, committed=false`，状态、调用数和 attempt 均未改变；待其他控制事务完成后顺序重放成功，只新增一个模型调用，没有把未提交控制请求算作第二次补正。

06:45:26 的[恢复后复核](check-9-66-after-recovery.json) 确认全部 12 项 running：9 项 P3、2 项 P4、1 项 P5；GDN 两分支并行，合计 13 个在途调用，当前无诊断。四项新回执尚未全部返回，运行恢复不代表相应 P3 已获批。
