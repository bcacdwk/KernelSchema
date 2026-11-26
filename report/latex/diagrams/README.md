# Kernel Schema 原生 TikZ 图

`diagram-01.tex` 至 `diagram-13.tex` 按 `report/基于 Schema 的 Kernel 表示与生成.md` 中 Mermaid 代码块的出现顺序对应。每个文件是可直接 `\input` 的 TikZ 图源，不含 `document` 环境，不依赖其他图源或图片。节点、边标签、分组标题及连接语义以原 Markdown 为准；仅调整布局、换行、颜色与线条。

## 主文接入

主文需提供以下宏包与库（本目录不修改主文）：

```tex
\usepackage{fontspec}
\usepackage{xeCJK}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,calc,backgrounds,shapes.geometric}
\setmainfont{Alibaba PuHuiTi}
\setCJKmainfont{Alibaba PuHuiTi}
```

从 `report/latex/` 编译的主文中可写：

```tex
\begin{figure}[htbp]
  \centering
  \input{diagrams/diagram-01.tex}
\end{figure}
```

使用 XeTeX 引擎；本次已通过 XeLaTeX 独立编译与渲染验证（TeX Live 2026）。tectonic 的默认 bundle 下载地址返回 HTTP 403，因此本次没有将 tectonic 编译标为通过。无需额外缩放，避免 `\resizebox` 缩小正文与标签。图中的颜色和样式各自定义。统一使用深色文字，蓝、青绿、琥珀色边框与淡色分组；普通节点为白底圆角卡片。换用其他字体后应重新检查尺寸。

## 文件对应

| 文件 | 原文位置 / 图内容 |
| --- | --- |
| `diagram-01.tex` | 第一章：未来 AI 闭环与当下 MaaS 刚需 |
| `diagram-02.tex` | 第三章：数据与表示、生成与迭代、交付与维护三个环节的挑战 |
| `diagram-03.tex` | 3.2：设计与实现耦合导致的路径坍缩 |
| `diagram-04.tex` | 4.1：人类 HPC 专家与编码 Agent 对比 |
| `diagram-05.tex` | 4.2：Kernel Schema 破局总蓝图 |
| `diagram-06.tex` | 第五章：规范、压缩、提取与知识库管线 |
| `diagram-07.tex` | 第五章 Design 区：三层递进设计 |
| `diagram-08.tex` | 5.3：P1–P6 蒸馏与三重防幻觉保障 |
| `diagram-09.tex` | 6.1：原生 AKA 与 Planner–Coder 解耦循环 |
| `diagram-10.tex` | 6.2：双起点与 Schema 设计遗传树 |
| `diagram-11.tex` | 6.3：Step 与 Dispatch 机制 |
| `diagram-12.tex` | 7.2：异步验证门禁与 Data/Bench 头尾闭环 |
| `diagram-13.tex` | 7.3：先验性能建模与近似保序剪枝 |
| `diagram-14.tex` | 6.2：设计空间与代码空间的邻域错位（直接在 LaTeX 正文中新增，不对应原 Markdown） |

## 独立测试

在项目根目录执行：

```sh
python3 report/latex/diagrams/_test/verify.py \
  --engine ~/.local/share/kernel-schema-texlive/bin/universal-darwin/xelatex
```

测试需要所选 XeTeX 引擎、Python 的 `PyMuPDF`（`fitz`）及已安装的 `Alibaba PuHuiTi`。不传 `--engine` 时默认调用 `tectonic`；可用 `--bundle /path/to/bundle` 指定其资源包。脚本创建 `_test/all-diagrams.tex`，按原生尺寸编译 13 页 `_test/all-diagrams.pdf`，将图片渲染到 `_test/page-01.png` 至 `_test/page-13.png`，并生成 `_test/verification.json`。另外逐图独立编译 `_test/diagram-01.pdf` 至 `_test/diagram-13.pdf`（各一页，紧边界加留白），方便拼图浏览。验证内容包括：

- 每图实际宽高是否不超过 168 × 180 mm。
- PDF 文本是否包含原 Mermaid 全部带引号的节点文字、连线标签与分组标题（仅忽略排版空白）。
- 字体名称、最小字号与栅格图片数量。
- 编译日志中的溢出、缺字和 LaTeX 字体警告。

`_test/` 仅保留验证脚本及最近验证摘要；运行脚本会重新生成独立 PDF、日志与预览，无需由主文加载。脚本直接读取原 Markdown。

## 本次验收结果

- 13 图全部通过 XeLaTeX 独立编译；总览 PDF 共 13 页，逐图 PDF 各 1 页。
- 所有原生宽度小于 168 mm，所有高度小于 180 mm；未整体缩放。
- 节点字号 9.5 pt，边标签 9 pt，组标题 10 pt；嵌入 Alibaba PuHuiTi 常规与粗体。
- 全部图均为原生 PDF 矢量线条及文字，栅格图片数为 0。
- 编译无溢出、缺字或字体警告；已渲染逐图检查连线、标签与卡片边界。
- 原 Mermaid 共 120 条文字项，源码与 PDF 两次核对均完整通过。`../qa/diagram-labels.json` 的 119 条全部保留；另外保留第 03 图在该清单中未收录的菱形节点（“遭遇实现层摩擦与环境扰动…”）。未修改该清单。
- 全部 79 条原始关系逐图核对，保持箭头方向、虚实/粗线语义、回环及分组关系。

| 图号 | 原生宽度 / mm | 原生高度 / mm |
| --- | ---: | ---: |
| 01 | 163.74 | 101.55 |
| 02 | 159.49 | 160.01 |
| 03 | 162.25 | 132.04 |
| 04 | 146.49 | 108.66 |
| 05 | 138.25 | 106.69 |
| 06 | 137.25 | 55.25 |
| 07 | 152.28 | 96.28 |
| 08 | 156.76 | 105.26 |
| 09 | 162.25 | 111.25 |
| 10 | 160.26 | 152.26 |
| 11 | 162.25 | 110.72 |
| 12 | 162.25 | 169.25 |
| 13 | 161.26 | 122.26 |
