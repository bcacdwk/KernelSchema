# 满额文本排版样例

直接阅读 [root.yaml](root.yaml)。这是固定三 step、两次 launch 的合成 attention 排版压力样例，不存在绝对最长：条目数未封顶，Stack/Wiki 等字符串也只有行数约束、没有字符上限。

最终 root 共 **475 物理行、41,433 个 Unicode 字符、41,455 UTF-8 字节**；最长物理行 **607 字符**（第 27 行）。160/160 个有字符预算的文本字段恰好到顶，其中 156 个同时用尽逻辑行数；4 个 REALIZATION.item 定位符各为 300 字符，保持合法的单行引用（允许最多 2 行）。四段 math 均为 1,000 字符 / 10 行，when 为 500 / 5，三段 thesis 与六段主要 choice/reason 均为 300 / 2。

按 DOC0–6 与 [FIELD_MAP](../FIELD_MAP.md) 逐路径核对，本结构适用的 **65/65 条可选路径**均有实际条目：

| 范围 | 已覆盖的可选内容 |
| --- | --- |
| DOC0–1 | source、Dispatch.id/when、Constant、nullable、range/default、Compute.Symbol、gap |
| DOC2 | Stack.Template，以及 Programming 的 Template / Virtual_isa / Machine_isa |
| DOC3 | Symbol 的 shape/dtype/retention/binding/value、Arithmetic 两项、Ledger 三项正文 |
| DOC4 | 三种 strategy、Partition.range、Invariant.offset、Iteration、Tile_reduction、Placement/member、group、count/stride、Refill、order/start_after、Handoff.initialize/ready/wait_at |
| DOC5 | Index 三种域、Event、Storage.member/View/slot_index/region、register、三类 Waiting；八 verbs、三模式、for/if |
| DOC6 | Require / Prefer / Forbid / Feedback，item 同时展示字符串与引用列表 |

root 禁止 Design.from，因此不适用；只交付一份 YAML。count 与 stride 分属不同 step，Symbol.value 与 binding 分属不同对象。未知 shape/dtype 不是新增字段，本例已明确属性，不制造未知值；Ledger 全为非空正文。长 item 与 range/default 都引用本文件实际声明的长名称，无尾空格或无意义字符补齐。

[budget_usage.csv](budget_usage.csv) 逐字段记录字符、逻辑行与限额。共 388 个字符串值：160 个有字符上限，12 个 Stack/Wiki 叶子仅限一行，其余 216 个是未设字符预算的来源、引用、动作或专门语法，保持正常长度；具名映射键亦无通用长度上限。

META_DATA.source 如实指向借鉴的既有 FA4 样例；testing.invalid 下的参考、资料与 Feedback evidence 均为明确的合成定位符，用于展示字段，不表示这些外部对象存在、硬件资料已绑定或有真实实验支持。设备指数近似的误差界也明确未知；这不是算法复原、数值验收或性能结论。

稳定的正式工具已完成下列命令，均返回 **0**；第二次报告 0 份需要排版，未使用跳过硬件、standalone 或绘图参数：

```bash
python3 -B schema_template/check_yaml.py schema_discussion/template_tooling/text_limit_demo/root.yaml
python3 -B schema_template/check_yaml.py --read-only schema_discussion/template_tooling/text_limit_demo/root.yaml
```

本机调用的 python3 为 `/opt/venv/python3.13.12/bin/python3`；[dispatch.md](dispatch.md) 仅提供正常目录检查必需的入口。
