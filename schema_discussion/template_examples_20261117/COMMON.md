# 四个 Template 案例的人工提取约定

当前工具基线为 `7a8030f`。本轮更新 `schema_template/examples/{fa4_prefill,fa4_decode,gdn_prefill,qsa_prefill}`，恢复后的固定材料见 [SOURCES.md](SOURCES.md)。每轮使用新的 high 代理，同时最多两位；主代理负责取舍、交接和验收。

## 阅读与来源

先读仓库 `AGENTS.md`、`EXTRACT_SCHEMA.md`、当前 `schema_docs/0.common.md`，再读相关 DOC 与 `schema_template/README.md`、`tools/README.md`。提取的选型和审阅职责参照 `schema_extract/phases/02_dispatch_draft.md`、`04_dispatch_plan.md`、`03_root.md`、`05_branch.md`、`06_final.md` 及 `schema_extract/REVIEW.md`；本轮手工推进，不运行 Runner、不制作机器阶段回执。

源码、合同、参考入口和已有测量是事实依据；旧 YAML 用于查线索和核对遗漏，不决定新的分支集合。原样恢复的四题快照是本轮来源，44 题批次的相近名称不能直接当作同一实现。源码引用使用已核实的固定 commit 与真实路径。

## 先选设计，再写完整 YAML

- 先对 host 选择器和关键实现做一次完整梳理，分别辨认参数、局部改变、已有组合与新的整体设计。选一份真实、便于理解的 root。
- 完整 YAML 只保留需要重新贯通主要计算、输出形成、工作归属或主协作的方案。相同机制的参数和局部变化记录到 `dispatch.md`，不按源码分支、分块取值或笛卡尔积建文件。
- 每个算子含 root 最多十份，目标是保留充分设计信息所需的少量代表，不以十份为配额。独立但未完整展开的候选如实说明，不假称局部修改。
- 每份 YAML 是一套已选的具体方案。算法本身的条件、尾块、空工作、领取与终止仍完整表达；不在 YAML 内藏配置选择器。
- 相同的完整 Design 区域直接 `from`；其他节点只继承 root 的 Contract。使用节点的接口、精度、范围、资源寿命与依赖必须成立。
- 分支选择先记在本次讨论目录的案例记录中，由下一位新代理对照源码独立审阅。审定后按该集合推进；只有新的源码事实或关键依据错误才重开选型。

## 写作与真实性

用当前 DOC 的字段和结构直接重写，删除被替代的旧 YAML 及其 PNG，更新存留设计的目录关系。`dispatch.md` 按 DOC0 的中文最小格式，变体直接定位当前完整基线及对象。

字符串只补充已有结构看不出的概念、决定和必要保证；不复述 shape、dtype、循环或公式，不塞入源码逐行说明，不创造缩写以压缩字数。用明确的 noun 区分角色动作，输入值、来源与关键依赖须可审阅。

不改 kernel，不虚构同步、初值、空工作、数值误差界或测量。提取保留源码实际做的工作，包括无效物化和已知限制。此轮是续修已有案例：既有 Device、Architecture、Stack、Execution、Feature 中可由匹配证据核实的事实应保留并转换为当前字段，不清空成 starter；未知 Wiki 入口按 DOC2 使用尖括号占位。不开展新的硬件调查，不拼接不匹配的设备、编译器或实验结论。

数学、工作调度、流水线和 Realization 应连接同一份具体设计。`from` 复用须逐区确认，目录文字不能替 YAML 补齐缺失的定义。

## 工具与验收

从仓库根目录运行，Python 使用 `/opt/venv/python3.13.12/bin/python3`：

```sh
python3 -B schema_template/check_yaml.py schema_template/examples/<case>/root.yaml
python3 -B schema_template/check_yaml.py schema_template/examples/<case>/root.yaml --read-only
python3 -B schema_template/check_yaml.py schema_template/examples/<case>/<id>.yaml --graph png --set <name>=<integer> --worktiles 1
```

实际执行使用上述固定 Python 路径。格式、引号、大小写与句尾交给工具；检查 warning 的语义是否保留，按 error 精简超长文本或修正真正的问题。正常化覆盖整个案例目录，目录内不放备用 YAML。

每份完整设计生成同名 PNG，记录其确切参数；样本要能检查关键推进、等待和环复用，展示缩小规模须明确其示意范围。默认检查与后续只读应通过，重复运行无待规范化差异。硬件占位确需 `--skip-hardware` 时逐项记录原因，不能把跳过硬件称为硬件验收。图未完成不能报告为成功。

不把合法源码设计改成错误的简化方案来绕工具。怀疑工具缺陷时记录最小复现、DOC 依据与影响，交主代理统一安排，避免并行修改公共工具。字段或语法需要实质变更则保留 issue，不自行增加。

独立审阅者核对分支价值、完整输入输出、数值路径、调度与动作顺序、准备/释放/协议依赖、来源身份和实际工具结果；发现可修的问题直接完善自己的案例。没有新变化或失败，不反复扩大测试。

## 文件职责

每位代理只编辑指定案例及自己的过程记录。案例作者不改其他案例、公共 README、正式规则、Extract 执行器或 `schema_wiki`。公共入口在案例稳定后统一更新，去掉延期与旧协议说明，给出现行用法、设计数量与真实验证边界。

过程文件留在本目录的 `<case>/`，交接应短：保留/合并的设计及理由、固定来源、实际修改、检查命令和结果、仍需审阅的事实。没有 GPU 重跑时不新增 GPU 验收结论，不提交 Git。
