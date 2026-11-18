# 真实时间与进度快照

采样：2026-11-18 02:29:53 UTC。来源为真实state.json、attempt.json、invocation.json、result.json及attempt_usage；不读取模型推理正文。本报告只读采样，不执行控制。

从首次FA4 run创建（11-17 13:32:26）至本采样，墙钟 **12.96小时**。两个算子并行运行，各run及各调用耗时不能相加当作用户等待时间。

**当前两个算子都完成P1、P2独立批准；FA4处P3独审，QSA处P3编写。均未ready、未published。** FA4后续仍需P3批准、P4、split_kv对应P5及P6；QSA仍需P3批准、P4、正式skip P5及P6。

| Run | 创建UTC | 首次CLI入口UTC | 状态/最高阶段 | attempt记录 | timeout | 同UUID续接调用 | 已结算调用账目 |
|---|---|---|---|---:|---:|---:|---:|
| FA4 r01 `8794f043` | 11-17 13:32:26 | 11-17 13:32:56 | paused / P1 worker | 2 | 2 | 0 | 0.754 h |
| QSA r01 `7c31f6e1` | 11-17 13:32:51 | 11-17 13:33:20 | paused / P1 worker | 2 | 2 | 0 | 0.753 h |
| FA4 r02 `a9a16cd1` | 11-17 14:46:11 | 11-17 14:46:48 | termination_unconfirmed / P2 worker | 6 | 0 | 2 | 2.707 h |
| QSA r02 `388e08c1` | 11-17 14:47:01 | 11-17 14:47:32 | invalid_receipt / P1 worker | 2 | 1 | 0 | 2.088 h |
| QSA r03 `f1a26608` | 11-17 17:13:36 | 11-17 17:18:50 | blocked / P2 reviewer | 5 | 0 | 0 | 3.575 h |
| FA4 r04 `ebe745a0` | 11-17 18:03:50 | 11-17 18:05:37 | running / P3 reviewer | 12 | 1 | 4 | 8.038 h |
| QSA r05 `dee63d48` | 11-17 21:41:53 | 11-17 21:42:44 | running / P3 worker | 10 | 0 | 5 | 4.206 h |

共 **7个run实例：FA4 3个、QSA 4个**；相对最初两个run，FA4重建2次、QSA重建3次，共5次重建。r04/r05是目录编号，不表示每个算子分别重来4/5次。同一run内resume、自动重试、退修和独审另计，不是整个流程重建。

总计39个attempt记录；其中当前2个在途、6个timeout。另2个历史attempt没有原生result：FA4 r02启动前拒绝且保留旧fence、FA4 r04孤儿P2独审后正式结清。不能把attempt记录数均称为成功模型调用。旧5个run已结算调用账目合计9.877小时；该数含重复劳动但不是额外串行墙钟，也不是全部无用工作，部分旧稿作为新run参考保留。

| 活跃run/阶段 | Worker/Reviewer调用数 | 状态 | 已结算Runner账目（分） | 有原生result的CLI时长（分） | 当前在途（分） |
|---|---:|---|---:|---:|---:|
| FA4 r04 P1 | 2 / 1 | approved | 59.16 | 58.58 | 0.00 |
| FA4 r04 P2 | 2 / 4 | approved | 217.00 | 138.62 | 0.00 |
| FA4 r04 P3 | 2 / 1 | reviewing | 206.12 | 205.74 | 13.73 |
| QSA r05 P1 | 1 / 1 | approved | 74.25 | 73.82 | 0.00 |
| QSA r05 P2 | 3 / 4 | approved | 178.12 | 176.90 | 0.00 |
| QSA r05 P3 | 1 / 0 | pending | 0.00 | 0.00 | 28.06 |

Runner账目包括启动/检查/结清开销；FA4 P2还包括孤儿调用至正式恢复结清的77.30分钟，该attempt没有原生result，因此原生CLI列有意不补造这段。当前在途按invocation.started_at至采样时间计算，尚未计入已结算账目。

| 活跃run | attempt | 阶段/角色 | 首次入口UTC | 账目时长（分） | 结果与重复原因 |
|---|---|---|---|---:|---|
| FA4 r04 | `db3d1015` | P1/worker | 11-17 18:05:37 | 31.06 | 原生complete，证据超旧12行拒绝 |
| FA4 r04 | `547fcd6b` | P1/worker | 11-17 18:39:15 | 5.69 | 同UUID修证据后complete |
| FA4 r04 | `bf8cadf6` | P1/reviewer | 11-17 18:45:01 | 22.41 | 独审批准 |
| FA4 r04 | `0c40f3b6` | P2/worker | 11-17 19:09:35 | 61.11 | complete后dispatch链接机器检查退回 |
| FA4 r04 | `9d74ca7d` | P2/worker | 11-17 20:10:50 | 10.58 | 同UUID修稿后complete |
| FA4 r04 | `0ef292dd` | P2/reviewer | 11-17 20:21:33 | 77.30 | 孤儿独审，无原生result；按正式恢复中断结清 |
| FA4 r04 | `e6336933` | P2/reviewer | 11-17 21:38:55 | 50.89 | 新独审原生complete，证据超旧12行拒绝 |
| FA4 r04 | `6cf40b72` | P2/reviewer | 11-17 22:32:08 | 10.03 | 同UUID修证据后approval handoff超限 |
| FA4 r04 | `57cf9b8a` | P2/reviewer | 11-17 22:42:23 | 7.08 | 自动receipt_repair后合法批准 |
| FA4 r04 | `fe7ec14b` | P3/worker | 11-17 22:49:38 | 120.22 | 7200秒timeout；规范化保稿 |
| FA4 r04 | `6d26c35c` | P3/worker | 11-18 00:50:07 | 85.89 | 新会话沿保稿complete，进入独审 |
| FA4 r04 | `ffecb3e9` | P3/reviewer | 11-18 02:16:09 | 13.73（在途） | 在途独立reviewer |
| QSA r05 | `18654451` | P1/worker | 11-17 21:42:44 | 34.79 | worker合法complete |
| QSA r05 | `802104c8` | P1/reviewer | 11-17 22:17:35 | 39.46 | 独审批准 |
| QSA r05 | `70533174` | P2/worker | 11-17 22:59:19 | 58.24 | 原生complete，证据超旧12行拒绝 |
| QSA r05 | `51fc0c10` | P2/worker | 11-17 23:59:36 | 10.91 | 同UUID修证据后complete |
| QSA r05 | `f4bc3e85` | P2/reviewer | 11-18 00:10:39 | 45.96 | 拟revise，但证据超旧12行拒绝 |
| QSA r05 | `fe0eebc3` | P2/reviewer | 11-18 00:58:17 | 15.00 | 同UUID修证据后合法revise |
| QSA r05 | `8cee3182` | P2/worker | 11-18 01:13:24 | 12.17 | 同UUID修dispatch硬件占位去向后complete |
| QSA r05 | `3008b562` | P2/reviewer | 11-18 01:25:39 | 14.29 | 拟approve但handoff超限 |
| QSA r05 | `7c56ec73` | P2/reviewer | 11-18 01:40:08 | 21.55 | 自动receipt_repair后合法批准 |
| QSA r05 | `29d53d49` | P3/worker | 11-18 02:01:49 | 28.06（在途） | 在途worker |

重复的主要原因：初轮900/1800秒调用预算不够；旧工具32K输出上限/验收Python误识JSON/无效回执后的UUID登记与维护恢复缺陷；旧12行证据规则和approval handoff长度导致多次重提；一处实际dispatch退修；FA4 P3首调用7200秒耗尽后保稿续写。源和schema字段没有因这些问题改写，新的run重查来源和独立批准；活跃run没有热升级。

当前产品已经修复相关协议/恢复问题并删除任意12行证据范围上限，但两活跃run继续使用各自冻结版本（仍有12行范围门槛）。此版本区别不能省略。Qwen qwen3.8-max[1m]/max的完整阅读、写稿、独审本身仍占主要在途时间；轮询间隔不会缩短这些调用。

本监督采用约45秒一次的只读巡检并向主代理简报，阶段变化即时记录；正常Runner自动调度下一角色、退修与timeout续写。本监督没有创建额外定时任务，也没有因本次统计暂停或重启run。
