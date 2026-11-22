# 不改 EXP 实现时的版本消费与交付边界审查

日期：2026-11-20。状态：只读源码审查及计划修订建议；不是实现、安装或兼容性验收记录。

范围：仅新增本报告。已读 KS/EXP 根 `AGENTS.md`；AKA 根未发现适用 `AGENTS.md`。先复用 `research/r1_aka_inventory.md`、`research/r2_aka_lifecycle.md` 的现状结论，再核对下列少量当前源码。没有运行 EXP 的 verify、doctor、setup、plan、capture、readiness、finalize、launch 或测试命令，没有写 EXP/AKA、Git 配置或远端，也没有触碰 KS 隔离工作树。EXP 部分源码有既有活动修改，本文引用的是本轮读到的工作树事实，不把这些内容宣称为已提交发行功能。

证据前缀：KS=`/home/shaohanyong.shy/KernelSchema`，AKA=`/home/shaohanyong.shy/atrex-kernel-agent`，EXP=`/home/shaohanyong.shy/AKA_experiment`。

## 1. 审查结论

**本次交付应收敛为 KS 独立产品及 AKA 内的可选 Schema 模块。原 X1 不应继续作为 EXP 开发单元，更不应阻止 KS/AKA 开工或独立发行。**

AKA 固定 KS 子模块、提供自己的安装/资源取得入口，并验证独立 AKA 的选择、资料、运行及恢复，足以构成这个产品边界。原 X1 中属于产品的安装、资源闭包和固定版本证据，可归并 A3/A7；EXP 的物化目录、来源账本、launcher、readiness、finalize、新 capture 和实验策略改造应退出当前计划。

同时，**“AKA 加了 KS 子模块，所以当前 EXP 更新 AKA 后自动可用”不符合现状**。当前 EXP 的 AKA 是生成式普通文件 vendor；它排除 gitlink，且另有固定安装、选择、导入与快照约束。源码更新、KS 可获得、可选包已安装、Schema workflow 被选中、EXP 冷启动/恢复可复现是不同事实，不能相互代签。

这些问题限制的是“当前 EXP 已兼容 Schema-on”的支持声明，不是 KS/AKA 的技术开工条件。无需通过新增 EXP 工程包解决它们。

## 2. 核实到的版本消费事实

| 环节 | 当前代码证据 | 对本次边界的含义 |
| --- | --- | --- |
| 固定 AKA 来源 | EXP:`control/aka-source.json:3` 记录上游 URL，`:5` 固定 `bf5557ac04f6ff9988901d511cf25e1cbe61a1bc`，`:7` 使用 `aka-ordinary-files-v1`。只读 `git --no-optional-locks ls-files --stage` 显示 `assets/aka/src/aka/cli.py` 与 Light 包 metadata 都是 `100644` | `assets/aka` 不是可直接 `git pull` 的嵌套 AKA checkout。独立 AKA 分支更新不会自行更新这份 vendor |
| 导出时丢弃子模块内容 | EXP:`tools/lib/aka_vendor.py:70` 从固定 Git tree 导出；`:93` 遇 `160000` 仅记 commit/URL/`materialized=False` 后跳过；`:98` 跳过 `.gitmodules`；`:150` 校验排除记录必须保持 `materialized is False` | 即使上游 AKA 工作树已递归初始化 KS，当前投影仍读 Git tree，不会把其工作树内容带进 vendor |
| vendor 不是随意补文件的资源目录 | EXP:`tools/lib/aka_vendor.py:181` 递归清点普通文件，`:189` 拒绝 `.git`/缓存目录，`:221` 比较整个 vendor，`:230` 要求差异与台账完全一致 | 不能把 KS checkout/下载缓存临时塞入 `assets/aka` 再宣称完成现有消费合同；这也不是本次应该执行的动作 |
| 安装集合固定 | EXP:`tools/lib/aka_dev_setup.py:16` 只列 framework/legacy/light；`:20` 从 vendor 的 builder/metadata 构建；`:38` 安装时使用 `--no-index --no-deps` | AKA 增加 `packages/atrex-aka-schema`、新增构建 owner，或只声明新依赖，都不会让 EXP 的既有 setup 自动安装该包/KS |
| 实际 Python 来源固定 | EXP:`tools/lib/environment_entry.py:8` 选已记录解释器；`:13` 设置 bridge 和 `assets/aka/src`；EXP:`tools/python/aka_experiment/runtime.py:175` 校验 AKA 模块来自 canonical vendor | 在独立 AKA 目录更新、安装或初始化 KS，不会自行切换 EXP 子进程的 AKA 来源 |
| 包及入口也被校验 | EXP:`tools/python/aka_experiment/runtime.py:163` 比较全部已安装 distribution 集合；`:190` 校验三包 metadata/版本/入口；`:216` 拒绝三包之外或重复的 `aka.commands`/`aka.tokens` | 外部装好新 schema 包不是现有 EXP 环境已兼容的证据；若新增包贡献这些入口，会直接碰到当前检查。其他包也不能绕过既有环境身份 |
| 启动选择固定 | EXP:`tools/python/aka_experiment/launcher.py:102` 只接受 `light-sandbox`；`:149` 从 vendor 的 Light composition 加载；`:154` 替换 Agent 为 EXP launcher；`:81` 始终叠 `SchemaAgent(DevelopmentAgent(...))` | 新增独立 AKA Schema profile/provider，并不意味着 EXP 自动选择它。现有 `SchemaAgent` 是原静态材料适配，不等于新的固定/动态 Schema backend |
| 当前正式 pilot 约束 | EXP:`tools/python/aka_experiment/launcher.py:480` 要求 `matrix.schema_enabled is False`；`:513` 生成 `condition=control`、`schema_enabled=False` | 当前 pilot 不是新 Schema-on 功能的现成入口。此事实仅划定未支持范围，不要求修改该 pilot |
| EXP 快照并非整个仓库或所有已下载资源 | EXP:`tools/python/aka_experiment/runtime.py:340` 只列固定源码/资源树；`:342` 只列固定文件；`:367` 只补三包 metadata；`:378` 要求无别名的普通文件 | AKA 的 `optional/KernelSchema`、新包 metadata 或独立 `assets/kernel-schema` 不在当前默认捕获集合；外部安装能导入，也不能据此宣布 EXP 冷启动/捕获闭包已完成 |

上述事实相互独立。只解决 gitlink 被排除这一项，不会自动解决安装、模块选择和捕获。

## 3. AKA 当前包装能力及应承担的交付

AKA 已有分包和独立构建入口，不需要借 EXP 才能发行：

- AKA:`tools/build_distributions.py:20` 定义每个 distribution 的源码 owner；`:29` 按 owner 收集源码；`:80` 从对应 metadata 构建 sdist/wheel。现有列表没有 Schema 包，构建也不会自动收集根目录外的 KS 子模块。
- AKA:`packages/atrex-aka-light/pyproject.toml:10` 固定依赖 legacy；`:18` 只收集 `aka.workflow.light`。因此新 Schema 模块的 ownership、metadata、package data 与 KS 依赖必须由 AKA 自己明确接入。
- AKA:`docs/quickstart.md:21` 已说明从 matching checkout 安装；`:33` 要求三包和资源 checkout 同 revision，即使 metadata 版本字符串未变，代码变化仍需重装。子模块初始化只是源码取得步骤，不是 Python 安装完成。
- AKA:`docs/quickstart.md:17` 要求只初始化选中的参考工程。首发只需取得 KS 和实际选中资料，不应为了 Schema-on 递归获取所有无关第三方仓库。
- AKA:`src/aka/bootstrap/continuation.py:67` 冻结实现与选中资源，`:161` 恢复时重新 capture 并比对。把 KS 资源声明纳入 AKA 自己的选择/恢复身份，属于 A3/A5/A7 的产品职责；它不要求开发 EXP 的 capture。

建议在 A3/A7 固定以下最小交付：

1. **A3 安装接口：**AKA 在评审过的 KS 提交上建立可选 gitlink；只有 Schema-on 才要求取得并安装 KS/Schema 包。明确 KS 核心、选中规则/方法/资料、Schema 包及 AKA 原包如何安装和定位；缺项给出可执行的安装说明。off 不初始化、不导入、不校验 KS，也不触发网络取得。
2. **A3 资源与包闭包：**新增 Schema 包的 owner/metadata/资源规则，明确使用安装包资源还是显式固定 checkout，避免从开发者 cwd 或临时缓存猜路径。若提供 AKA 自有安装/获取命令，其内容只处理 AKA/KS 产品依赖；不得通过该命令改写 EXP 环境、配置或捕获。
3. **A7 固定发行：**记录 upstream/fork/KS 提交及实际包、规则、资料和所选 profile 身份，保留许可与可访问来源；在新目录、非原 cwd、专用环境验证 off 和所声明的 Schema-on 入口，以及 fresh-child 的实际来源/资源。恢复验证使用 AKA 自己的固定运行产物。
4. **A7 如实声明支持：**独立 AKA fixed/dynamic 按各自真实门签收；EXP 的 vendor 更新和 Schema-on 消费没有验收时写“未验证/不在本次支持声明内”。不能将该缺项反向加为 A7.fixed 前置。

KS 尚无可交付固定提交时，A3 可先消费约定接口或显式开发源码进行工程验证；“正式固定 gitlink 发行”留到 A7。不能拿 KS 当前脏目录冒充固定版本，也不需为此停下所有 K/A 工作包。

## 4. 递归 checkout 或 AKA 自有获取命令能解决到哪里

| 路线 | 可成立的结论 | 不能推导的结论 |
| --- | --- | --- |
| 新的独立 AKA checkout，固定 fork SHA，显式初始化 KS gitlink，再按 AKA 的安装入口装包/资源 | 在 A3/A7 实现并验证后，足以交付独立 AKA Schema 功能。初始化可限定 `optional/KernelSchema` 及真正需要的嵌套资料；无需全仓第三方递归 | 仅执行 `git pull` 或 `git submodule update` 就已完成安装/运行；当前 EXP 自动改用此 checkout |
| AKA 自有安装/资源取得命令，从固定 KS commit/发行产物取得依赖 | 可以完全在 AKA/KS 范围内实现产品闭包。若要支持没有 `.git` 的源码发行，需要 AKA 自有可读的固定来源记录或打包产物，不能只依赖不存在的 gitlink 工作区 | 当前 EXP 已安装对应包、选中新 workflow，或会捕获下载位置 |
| 用现有 EXP vendor 工具更新到新的固定 AKA fork | 未来由 EXP 维护者作为版本升级操作，可能取得 AKA 普通源码改动；现有工具已支持 `--repository` 和固定 commit，无需本次另开发导入器 | KS gitlink 内容会一起导出，或旧 overlay 无冲突，新环境/旧 capture 自动兼容 |
| 在独立环境运行新的 AKA，然后由任意外层程序启动多个独立 AKA 实例 | 外层消费 AKA 是合理目标；本次可提供稳定 AKA 命令/API 供外部调用 | 当前 EXP 已具有任意外部 AKA 启动入口。现有 launcher 源码不支持这种自动推断 |

第二条可以改善产品的可分发性，但不必为了当前 EXP 特制新的下载系统。首发选择已能验证的一条独立 AKA 安装路径即可；若公开源码发行本来要求无 `.git` 安装，则在 A3/A7 内完成对应产品测试。禁止将网络补齐资源放到被冻结运行的恢复过程中，以掩盖版本或资源缺项。

## 5. 原 X1 的逐项去向

| 原 X1 内容 | 本轮建议去向 |
| --- | --- |
| fork/KS 固定版本、可访问来源、许可、发行证据 | 归 A7；实现级依赖接口归 A3 |
| KS 内容取得、字节/模式/规则/资料闭包 | 归 A3 的 AKA 自有安装/资源接口，A7 用独立安装验收；删除指定 EXP 目录的设计 |
| metadata/resources、fresh-child 实际模块来源 | 归 A3/A7 的独立 AKA 验证；不复用“EXP 已验证”的名称 |
| 产品只有 KS/AKA 一份可编辑来源 | 保留为两仓维护原则；EXP 若后续更新，只消费固定产品版本 |
| 固定 AKA 普通文件 vendor 升级 | 仅在维护说明写现有 EXP 是生成式 vendor，日后由其维护流程评审升级；本次不运行、不改文件，不设工程 gate |
| 修改 `tools/lib/aka_vendor.py`，新增 `assets/kernel-schema`/`control/schema-source.json` | 从当前实施要求删除。这是 EXP 新物化/台账方案，不是 AKA 可选模块所需 |
| 修改 EXP setup/environment/launcher/readiness/finalize | 从当前实施要求删除，不改名并入 A3/A7 |
| 新 EXP capture、KS manifest 集成、冷启动/恢复与旧 reader 迁移验证 | 保留为“当前 EXP 消费尚未验证”的事实；取消其开发包/门槛，不在本次设计新的 capture |
| 新 launch 改用新模块、旧 SchemaAgent 只服务旧捕获 | 删除强制迁移指令。当前没有兼容证据，也没有本次 EXP 改造授权 |
| 回退/保留旧运行 | 保留有限维护原则：不修改活动运行和旧冻结材料；只在有具体外部升级时使用原维护流程。无需预制 EXP 新回退工程 |
| X1 与 X2.protocol/run 的依赖、EXP-P1 发布门、B10/B11 的 EXP 派工 | 从本次 K/A 依赖图和派工删除；不能换名为“A7 内部 EXP 集成”继续实施 |

`05_work_packages.md:158` 当前把 vendor、KS 目录、来源账本、setup/launcher/readiness/finalize 全列为 X1 改动；这是需要删除的实际扩域，不只是措辞调整。`04_implementation_plan.md:82` 的强制单独物化和 `:83` 的新旧 Agent 切换也需撤出规范。

V14 应保留 A3 安装接口、A7.fixed 独立发行及 A7.pair 对接证据，删除 EXP manifest/capture 作为本次完成条件。公开材料表中可以保留消费者兼容性状态，但不能继续要求交付 EXP 新 capture。X2 实验设计不属于本报告技术评审范围；在用户不改 EXP 的边界下，不应由 X1 留下其前置链。

## 6. 开工阻塞、发行阻塞与范围外事项

| 事项 | 对本次工作的影响 |
| --- | --- |
| 独立 AKA 需要明确 Schema 包 ownership、KS 安装接口和资源来源 | 是 A3 的必要实现内容，未完成会阻止对应安装签收；不阻止先做 KS 合同、A1/A2 或安装接口开发 |
| KS 没有评审后的可交付提交，或资源/许可/来源闭包未完成 | 阻止 A7 正式固定发行；不是所有工程包的开工门 |
| AKA 的 off、固定请求/恢复或真实任务资格未通过 | 分别限制 A3/A5/A7 对应支持门；必须留真实证据，不能由 EXP 缺项掩盖 |
| EXP 投影排除 gitlink，三包装配及入口限制，固定 launcher 与快照集合 | 不阻止 K/A；只说明当前 EXP Schema-on 未验，且现有代码对部分方案已有明确不相容点 |
| 当前 EXP 脏修改、活跃任务、旧 capture 及正式 pilot 设置 | 不在本次实施范围，也不是 K/A 外部前置；本轮保持只读 |
| 尚未执行 EXP 固定来源升级、冷启动或真实消费 | 不在本次完成声明内；无需扩展计划去实现，也不能写“已兼容” |

可用于当前计划的一段准确表述：

> 本次只在 KS 和 AKA 实现并交付产品。KS 作为 AKA 显式启用的固定可选子模块，安装、资源取得和独立验证由 AKA 自有入口承担，A3/A7 负责闭合。EXP 是外部消费者，本次不改其实现、配置或捕获。当前 EXP 使用排除 gitlink 的普通文件 vendor；后续是否仅通过其既有版本升级操作即可消费，应以实际来源、安装、选择和运行验证为准，本次不声明自动兼容，也不以 EXP 消费缺项阻止 KS/AKA 开工或独立发行。

## 7. 验证范围

本报告的证据来自文本阅读、定向 `rg`、带行号源码读取及一次指定路径的只读 Git 索引查询。没有运行任何产品/实验工作流或安装测试；所有未来安装、fresh-child、Schema-on、恢复和真实模型/GPU 结果均未在本轮验证。只读审查足以否定“gitlink 自动被现有 EXP vendor 携带”的假设，也足以确定本次无需 EXP 开发包；它不足以宣布某个尚未实现的发行配对已可运行。
