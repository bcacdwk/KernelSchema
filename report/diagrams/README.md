# Kernel Schema 原生 TikZ 图

每个文件是可直接 `\input` 的 TikZ 图源，不含 `document` 环境，不依赖其他图源或图片。正文通过 `DiagramBlock` 环境引入，按版心宽度自适应。

图中的颜色和样式各自定义：统一使用深色文字，蓝、青绿、琥珀色边框与淡色分组；普通节点为白底圆角卡片，节点字号 9.5 pt，边标签 9 pt，组标题 10 pt。原生宽度控制在 168 mm 以内，避免整体缩放后字号变小。换用其他字体后应重新检查尺寸。

01～13 号图最初由原 Markdown 中的 Mermaid 代码块转写而来，早期的逐图核对脚本与记录已移至 `../old/diagrams/`。`diagram-11.tex`（Step 与 Dispatch 机制）已从正文移除，存档于同一目录。

## 文件对应

| 文件 | 位置 / 图内容 |
| --- | --- |
| `diagram-01.tex` | 第一章：当下 MaaS 需求与未来可验证闭环 |
| `diagram-02.tex` | 第三章：数据与表示、生成与迭代、交付与维护三个环节的挑战 |
| `diagram-03.tex` | 3.2：设计与实现耦合导致的路径坍缩 |
| `diagram-04.tex` | 4.1：人类 HPC 专家与编码 Agent 对比 |
| `diagram-05.tex` | 4.2：Kernel Schema 总体思路 |
| `diagram-06.tex` | 第五章：规范、压缩、提取与知识库管线 |
| `diagram-07.tex` | 5.1：Design 区三层递进设计 |
| `diagram-08.tex` | 5.3：P1–P6 蒸馏与防幻觉校验 |
| `diagram-09.tex` | 6.1：原生 AKA 与 Planner–Coder 解耦循环 |
| `diagram-10.tex` | 6.3：双起点与 Schema 设计遗传树 |
| `diagram-12.tex` | 7.2：异步验证门禁与 Data/Bench 头尾闭环 |
| `diagram-13.tex` | 7.3：先验性能建模与近似保序剪枝 |
| `diagram-14.tex` | 6.2：设计空间与代码空间的邻域错位 |
| `diagram-15.tex` | 8.4：从 Kernel 到自动化科研的 Infra 自进化阶梯 |
