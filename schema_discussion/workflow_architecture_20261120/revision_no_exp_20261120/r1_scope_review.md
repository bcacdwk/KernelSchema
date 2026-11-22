# 本轮仅 KS / AKA：范围审查与可执行修改清单

日期：2026-11-20。独立审查，只读正式计划及必要来源；只写本报告，未改实现、正式计划或 EXP，未运行模型/GPU、安装、迁移、提交或推送。已读 KS AGENTS.md；读取 EXP 工具前另读其 AGENTS.md。D1–D3 保持。

下列 P/W/C/V 分别是 `04_implementation_plan.md`、`05_work_packages.md`、`planning/r5_contract_execution_review.md`、`planning/r5_validation_delivery_plan.md`。行号按本目录 `before/` 中对应快照；当前文件同内容。历史 research/R4 不作为现行规范。

## 1. 结论与原计划为何纳入 EXP

本次实施应收敛到 **K1–K8、A1–A7 共 15 单元**。删除 X1/X2、EXP-P1 和所有 EXP 工程/实验验收门；不把 X2 改名成 AKA 的实验包。EXP 只作为外部消费者或必要补丁的只读来源出现，不是第三个实施仓。

原来的扩展有现实原因，但不是 KS/AKA 产品必须承担的工作：

- 初始简报同时要求考察三个仓库、指出 EXP 用于多个 AKA 集合实验，并倾向由 EXP 消费适配 fork；同时明确要避免功能开发与实验适配混在一起（`00_user_brief.md:9,19`）。后续计划把“未来消费关系”扩大成当前迁移和 B/F/D 验证义务（P:25–35、72、82–85；W:153–166）。
- EXP 的现有消费确实不是递归 checkout。此次只读复核 [aka_vendor.py](/home/shaohanyong.shy/AKA_experiment/tools/lib/aka_vendor.py:93)：mode=160000 被记录为未物化后跳过，`.gitmodules` 也跳过（93–99 行）。这解释了 X1 的 KS 另行物化方案；它是 EXP 的特殊消费问题，不是 Schema 可选模块必须建在 EXP 的理由。
- B/F/D 9 run 是比较基准、固定 Schema、动态策略的研究方案（V:189–219），没有承担独立 fixed、真实 design/optimize、动态恢复或复杂 F1 的唯一产品证据。既有 K/A gate 已分别承担这些义务。

因此，用户本次缩小范围后不需要重新决策 D1–D3，也无需再询问是否删除 EXP 工作。

## 2. 必删与保留边界

| 项目及证据 | 直接修改 |
| --- | --- |
| P:1、9、11、25–35 | 标题/规范职责改为双仓、15 单元、无 pilot。职责表只列 KS 与 AKA；依赖图主线为 AKA 可选固定 KS、KS 可调用外部已安装 adapter。EXP 如需出现，仅在外部使用说明中举例。 |
| P:72；W:50–53、153–166 | 删除 EXP-P1、X1.engineering/consume、X2.protocol/run 及完整 X1/X2 描述。不得留下失去 owner 的 vendor、物化、capture、预登记或分析交付。 |
| W:1–5、10–11、55、181–189、194–196 | 全部计数改 15；仓库/基线改 KS/AKA。删 X 依赖、X 专属 `.protocol/.consume/.run` 标签说明及 owner；V08/V09 的 A6 消费保留，V12/V14 保留 A3/A7，公开材料只剩 K/A owner。 |
| P:80–85 | 基线/隔离约束只对两个实施仓；继续保护既有脏树。必要 EXP 补丁只能逐文件只读追源后在 AKA 实现，不整批迁 EXP 根提交。不再安排 EXP vendor/KS 物化、旧 SchemaAgent 切换、新 capture、回退或实验。通用固定配对、旧资源保留、off/恢复回归仍保留。 |
| V:149（V14） | 改为“AKA 固定源码发行与 KS 配对”。删 EXP vendor/KS manifest、EXP fresh-child 来源、新 capture/旧 reader 恢复。保留 A3 的实际安装/资源/off、A7.fixed 独立发行、A7.pair 双后端各自真实资格；不删 V14 整行。 |
| V:170 | 改为 AKA 的 upstream/fork/KS/adapter/profile 来源与配对证据，不交 EXP 策略、vendor 或新 capture 身份。 |
| V:185–219、227 | 移出当前规范：三组处理、对齐/预登记、跨候选实验选择器、9 run、运行顺序、8 calls/6 jobs/3 e/1h 建议、实验总账/摊销/统计列/全样本分析。历史快照已保存，不另造一个当前必须执行的“后续研究包”。 |
| P:11、85；W:4；C:4、135；V:1、3、5、77；README:3、10–14、23、32、45 | 清理 pilot/三仓/17 单元/12 批的现行摘要与链接。C:135 保留历史初始化费用与当前费用分列，仅去实验成本链接。历史经过只追加新记录，不改写旧发生事实。 |

“capture”不是一律删除的关键词。V:167 的 KS demo `evidence/<capture-id>/` 只是证据目录；V:124 的 CUDA capture 限制也是案例事实。两者不是 EXP 工程，应保留。

## 3. 从 pilot 抽离的产品规则：只保留实际产品所需部分

### 3.1 预算与费用不能随 X2 删除

C:123–134 已独立完整定义有限上限、同父累计、全部实际收费接点、收据去重、未知费用、暂停/fence、单 owner 与跨进程授权；这些由 K4/A5/A6 及 V08 验证。V:196 的组间额度对齐可删除，不能据此删上述任何产品能力。也不将 pilot 的建议上限变为所有 KS/AKA run 的硬编码默认。

C:135 保留“来源中的历史 design/extract 初始化费用单列”；不要求产品构建 B/F/D 实验总账或计算 `C_init/m+C_use`。产品继续准确返回原始用量，研究统计留给未来外部消费者。

### 3.2 选择器不要整包迁到 AKA

V:198–208 的“按全 run 首次登记候选序号选最大合格候选”是 pilot 交付政策，应随 X2 移出。当前产品合同只必需区分当前请求的 latest/qualified、历史所属 R 与 native G（C:158–175）。不能为了删除 EXP，反而在 A6 添加统一跨 R 候选登记、B 组支持和实验终态选择服务。

V09 目前把产品恢复和实验选择混在一行（V:144、154；W:182）。建议改名为“写作/新 R/恢复与资格归属”，删除“三条 pilot 候选轨迹”和独立选择器耗时不变性测试，改为以下已有产品合同的三条有界 CPU 验证，由 K5/A6 消费：

1. 新 R 已准入且其 C 合格，返回新 R 的当前 qualified；较慢不妨碍硬条件合格，不按 G/晋升替代。
2. 新 R 尚未完成，当前状态仍 partial/blocked；旧 `(R,C)` 可保留并明确标历史，不能伪装成当前 R qualified。
3. 旧 Q/R 迟到收据只吸收至原 operation/资格/账户一次，不改写当前 admitted request，不把旧 pass 挂到新 R，不绕过未结算 fence。

这些是请求身份、恢复与结果投影，不需要实验候选优选规则。若正式修订把第 2、3 条补到 C:161 附近，应仅补明上述归属规则，不扩公共结果为研究统计 API。

### 3.3 extract 消费模式仍须明确

V:221–225 的“原码复验 / 显式 seed 续改 / 纯 Schema 重建”是公开使用的信息边界，须从 pilot 节移到材料或公开使用节，给独立 `#consumption` 锚点；同步 P:56、V:77。保留 verify_only、准确 seed、资料 allowlist 及结论限制。它们不要求增加对照组、9 run 或重建实验。

## 4. A3/A7 本就应该完成 submodule 与安装交付

现行 A3 的 fixed 源码接口、可选安装/off、资源投递（W:40–41、123–127）和 A7 的固定配对、独立安装/公开取得/许可（W:48–49、147–151）已经拥有该职责。更早 MIG-03 也将固定 KS gitlink、安装、资源配对单独列为 AKA 交付，而 MIG-04 才是 EXP 另行物化（历史 `planning/r4_aka_integration_plan.md:302–317`）。去 EXP 不产生新安装包。

建议把现行偏概括的 A3/A7 补成可验收的要求：

- **A3**：在 AKA 内固定可选 KS submodule 及实际 optional schema 包；schema-on 的文档化安装只初始化所需固定子模块并安装所需 KS 核心/AKA 模块与资源。默认安装/off 不要求 KS 存在、初始化、扫描或导入；on 缺固定资源给明确诊断。安装路径不能依赖作者 cwd 或脏 KS 工作树。
- **A7.fixed**：用独立目录、固定 upstream/fork/KS 配对及公开可取得来源验收源码 checkout/editable 安装、模块 metadata/规则/Template/角色资源和实际可调用入口；保持现行无 wheel 承诺。保存 license/source/evidence 索引及可操作安装说明。
- **A7.pair**：继续消费 A2 的统一 fixture 与双方各自真实记录，只控制 KS↔AKA 跨后端声明，不等 EXP。A7.fixed 不因等待 pair/K6/A6 被阻塞（P:69–71；W:48–49）。

外部消费者（包括以后 EXP）应消费这一文档化的固定 AKA+可选 KS 交付。可在 AKA 使用说明中写“普通文件 vendor 不自动带入 gitlink；使用者须取得声明的 KS 版本/资源”，但不实现 `EXP:assets/kernel-schema`、`control/schema-source.json` 或 EXP setup/launcher 捕获工程。

**真实风险必须保留：** 当前 EXP vendor 跳过 gitlink，所以“只更新 AKA SHA 就自然具备 schema-on”不成立。此次应准确声明未验证/未交付 EXP 集成；这既不阻塞 KS/AKA 产品，也不成为要求用户现在授权 EXP 改造的理由。

## 5. B09–B12 及隐性尾部依赖的具体改法

前 B01–B08 的技术主线保留；B08 “不等 EXP”可改成独立签收范围，不暗示本轮稍后还有 EXP 门。

| 原批次（W:219–222） | 本次修改建议 |
| --- | --- |
| B09 | 保留 A6.cpu/real。删除 X2.protocol；第二槽可整理 A7 已有职责内的 optional-module 安装/固定消费说明，或空闲，不能安排改名后的预登记。 |
| B10 | 保留独立审查 A6、KS↔AKA 对接与 A7.pair；明确“跨仓”仅指这两个仓。删除 X1.engineering；第二槽只整理现有 K6/A7 支持/来源/证据文件，最终配对在审查修复后固定。 |
| B11 | 删除 X1.consume→X2.run 和“等全部试点结果再定稿”依赖。其原说明整理并入 B10/现有 owner，不新增技术包。 |
| B12 | 原最终源码/版本/证据/未完成项检查保留并前移为新的 B11。最终依据是 K/A gate，无实验报告完成门。 |

因此基准表可直接改成 **11 批**；这仍是有前置条件的派工基线，不是工期/轮数承诺。W:206–207、224、227–228 同步去试点、EXP 和 X1 vendor 例子，固定提交/快照交接与防并发编辑继续保留。README 的批次数同步。

## 6. 修订完成检查与剩余真风险

修订者用当前四正文、README 和追加历史核对：没有现行 X owner/EXP-P1/实验产物目录/9 run/EXP vendor 写路径；没有因删除 pilot 留下坏链接；15 单元依赖仍闭合；基线写入范围只含两仓；历史 before/research/R4 不被批量替换。

必须继续保留：KS 无 AKA 的独立实现与公开 demo；fixed 与准确跨 e 续写；真实 design/一次 optimize/合法 keep；FA4 早期预审及真实 F1；optional/off；Q/R/C 与 source/evaluator 身份；V0/C0/G/HEAD 分离；三 audit；同父预算和首 result 前恢复；两类 unknown；两后端各自 hook/模型/执行资格；实际源格式与许可/闭包缺项。它们都有 K/A owner，不是 EXP 删除项。

当前剩余工程风险仍是既定 spike/profile 的可用性、准确 native 生命周期接点、真实 evaluator/硬件、有限 fidelity、固定源码资源与公开许可，而非 EXP 是否跑过 9 run。**本次无需用户再次决策，也不启动实现或模型/GPU。**
