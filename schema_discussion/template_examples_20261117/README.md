# 四个 Template 案例更新

工作起点 `7a8030f`；共用约定见 [COMMON.md](COMMON.md)，固定来源见 [SOURCES.md](SOURCES.md)。本轮按现行 Extract 文字方法人工完成，没有运行 Runner/GPU。

| 案例 | 原→现行 YAML 数 | 当前结果 |
| --- | ---: | --- |
| [FA4 prefill](fa4_prefill/REVIEW.md) | 11→3 | 独立全文审阅完成；三份主图、N1、空 split 与 padding 检查通过 |
| [FA4 decode](fa4_decode/REVIEW.md) | 7→4 | 独立全文审阅完成；四图及不等长16成员检查通过，cluster 主图在 when 内 |
| [GDN prefill](gdn_prefill/REVIEW.md) | 2→1 | 独立全文审阅完成；6+2 主图和最小2 chunks 检查通过 |
| [QSA prefill](qsa_prefill/REVIEW.md) | 1→1 | 独立全文审阅完成；2×4局部协议示意及空 worker 检查通过，最终 PNG 来源已同步 |

四题原件已恢复至 `kernel_wiki/aka-four-problems-20261019`，470 文件与固定 Git tree 的内容及模式一致；44 题主批次索引检查通过。当前交付为九份 YAML/PNG、四份目录及案例 README，公共入口见 [Template 案例](../../schema_template/examples/README.md)。

本轮已完成。公共工具修复局部索引/窗口误拒，支持 Mapping.Member 已分配的单点数学坐标；追加 xhigh 审阅补齐局部同名索引遮蔽与固定参数联合检查，并精简实现，最终137项测试通过。九份主图与七组边界在最终工具上复核通过，现有 PNG 无需再次重绘。阶段记录见 [TOOL_FIXES.md](TOOL_FIXES.md)、[MEMBER_GRAPH.md](MEMBER_GRAPH.md)、[TOOLS_POLISH.md](TOOLS_POLISH.md)；最终交付与检查见 [FINAL.md](FINAL.md)，简短修改见 [CHANGELOG.md](CHANGELOG.md)。
