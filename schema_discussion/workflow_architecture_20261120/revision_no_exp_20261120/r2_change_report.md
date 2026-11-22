# 双仓范围修订：实际变更与检查

日期：2026-11-20。状态：正式文档已修改；不是产品实现或运行验收。已读 KS AGENTS.md、两份独立审查和四份现行规范；本代理未派生代理，未读取或修改 EXP 源码，未操作 Git、隔离工作树、安装、模型/GPU、提交或发布。01_history.md 由主代理维护，本代理未改。

## 1. 已落实的范围与运行目标

本轮修改 P=`04_implementation_plan.md`、W=`05_work_packages.md`、C=`planning/r5_contract_execution_review.md`、V=`planning/r5_validation_delivery_plan.md` 及两份 README，并新增本报告。此前 34→17 是已发生的四轮复审结果，历史不改；本轮进一步 **17→15**，只保留 K1–K8、A1–A7。

| 采纳或调整 | 实际落点 |
| --- | --- |
| 采纳两份审查的双仓边界 | P 删除第三仓职责、图中消费工程、发布门与维护写路径；四份规范只在 P 的一段范围外说明保留外部消费事实。W 删除 X1/X2 及四个子 gate，无 owner/依赖残留 |
| 安装与来源仍归原 A3/A7 | A3 明确固定可选 KS gitlink、包 ownership/metadata、实际安装/资源定位与 on/off switch；A7 在独立目录、非原 cwd 验证公开固定源码/许可、实际包资源及 fresh-child 来源。只交已验证的 AKA 自有 checkout/editable 路径，不新增下载平台、wheel 或全仓递归义务 |
| 删除实验工程，不改名塞进 AKA | V 删除 B/F/D、9 run、预登记、实验预算建议、候选全局排序选择器、组间统计/摊销；W 不新增替代包。原 X1 的指定物化目录、来源台账、setup/launcher/readiness/finalize、捕获/迁移/回退工程没有转移到 A 系列 |
| 保留产品预算与请求归属 | C 预算/恢复正文保留；历史初始化与当前费用分列，缺失不记零。C/V09 只澄清当前 R 与历史 qualified 分离、迟到原收据回原 operation/主体/账户去重、不改 admitted_request 或绕过 fence；不强加候选排序或新结果 API |
| extract 三模式从旧实验节独立出来 | V 新增 `#consumption`，保留原码 verify_only、显式准确 seed 续改、纯 Schema 重建的 allowlist 与结论限制；P/V 链接已改指新锚点 |
| 融入用户最新两条完整目标 | P 开头/收尾、K6/A3/A6/A7、V09/V13/V14 与 V 最终使用表、B06/B09/B11 明确 AKA switch 动态迭代及 KS 独立迭代；AKA fixed 仍可先签收，但不能冒充完整目标 |
| 补明已有 driver 的行动消费 | K6 在有限单 owner/同父预算内连起 design 或 extract 消费→implement→至多一次 optimize；合法 revise/同 R implement/verify 必须实际执行后续固定请求，keep 无行动合法结束。实际未发生分支用 CPU fixture 标明，不强迫真实修订或无限循环 |

消费审查的事实足以否定“AKA 新增子模块后，外部普通文件 vendor 只更新 SHA 就自动可用”；它不足以宣布尚未实现的安装配对可运行。因此本轮只保留[简明范围外边界](../04_implementation_plan.md#external-consumers)，不把该缺项变成 KS/AKA 前置。A3 工程可先用固定接口或显式开发来源，A7 正式发行必须绑定可取得的已审 KS 提交，不使用活动脏目录冒充固定版本。

D1–D3、单 owner 有限预算、首 result 前与跨进程定位恢复、准确源码/跨 e 续写、两类 unknown、三 audit、V0/C0/G/HEAD、off、Q/R/C 与 evaluator 身份、当前完整 Schema 的有限 fidelity、两后端各自真实资格、KS 独立链及真实 F1 均保留。Task 参数、比较器、计时、F1 范围和原栈要求未改。

## 2. 单元、派工与状态

- 单元：15 个，各有唯一描述和 owner；子 gate：34 个。
- 依赖：76 条显式及条件边，按含替代路径的保守图检查无环；AKA fixed 不依赖 KS 真实 demo、A6 或 A7.pair，KS 参考真实实现不依赖 AKA。
- 派工：B01–B11 共 11 个基准批次，覆盖全部子 gate。旧 B11 实验批次删除；旧 B12 最终检查前移为 B11；B09/B10 空余槽仅整理原 K6/A7 职责或空闲，不为数量新增包。
- 最终交付核对两条可运行路径及全部适用 K/A gate；真实 keep 正常，失败/缺项照实报告，不能以 fixed 或一边的 pass 代签另一门。
- 实际状态：技术单元全部未开始，实现派工暂停。AKA 本地与远端 `feat/schema-workflows-v1` 已在此前建立；KS 隔离工作树创建被用户中断、尚未确认完成。导航已改正“未做过远端操作”的过时暗示；本轮没有产品代码、模型/GPU、源码提交或发布。

## 3. 离线检查与真实局限

执行定向 `rg -n` 检查现行四规范和导航的 X/EXP/pilot/旧计数/旧锚点/写路径；再用一次性 `python3 - <<'PY'` 读取 Markdown，做以下断言：

| 检查 | 结果 |
| --- | --- |
| 唯一单元/子 gate 与引用 | K1–K8、A1–A7 共 15；34 个子 gate，显式及简写引用均闭合 |
| DAG 与独立门 | 76 条依赖含显式条件，拓扑检查通过；A7.fixed、KS 参考真实链的独立性通过 |
| 派工与验收 | B01–B11 唯一，34 个子 gate 均有批次；V01–V15 恰各一行，全部有适用 K/A owner |
| 范围清理 | 四规范无 X1/X2、EXP-P1、旧实验锚点/写路径/必交实验；EXP 仅一段范围外说明。保留的 capture 是 KS 证据目录或 CUDA 案例语义；统计 null 是产品用量事实 |
| 本地链接及锚点 | 当前四规范、主导航、本轮导航/报告共 7 份文件，94 条本地目标与锚点全部有效 |
| 保护摘要 | 两轮 before/baseline 全部匹配，决定页与主计划 D1–D3 原文不变；C 的 API/身份/源码/发布/设计变更/fidelity/恢复及 V 的 Task/adapter/F1 章节逐字节未变；修改前采集的 36 个非授权文件摘要逐一相同，01_history.md 不在本代理写入集合 |

检查只证明文档结构、引用与范围一致性，不证明安装、switch、迭代 driver、恢复或模型/GPU 已能运行；确切使用命令由对应实现 owner 落到 quickstart 并实际验证。原有风险仍是 native 生命周期/profile 接点、完整 evaluator/实际硬件、有限 fidelity、F1 原栈及来源/许可闭包。没有运行产品测试或模型/GPU，也没有再次向用户请求方向决策。
