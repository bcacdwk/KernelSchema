# 基于 Schema 的 Kernel 表示与生成 · LaTeX

## 快捷编译

```sh
cd /Users/hanyong/KernelSchema/report
latexmk
```

目录内的 `.latexmkrc` 默认使用 XeLaTeX，自动更新 YAML 高亮、重跑目录与交叉引用，然后将 `build/main.pdf` 同步为根目录的 **`基于 Schema 的 Kernel 表示与生成.pdf`**。没有修改时，不重复编译。

```sh
latexmk -pvc   # 监视正文、样式与图源等依赖，修改后自动重编译
latexmk -g    # 强制重编译（例如修改了高亮脚本后）
latexmk -c    # 清理编译辅助文件，保留 PDF
latexmk -C    # 清理 build 下全部编译产物；保留根目录的中文名分发 PDF
```

请先进入本目录再执行，确保 `.latexmkrc` 被读取。

## 编辑入口

- `body.tex`：正文与 YAML 代码块，直接修改此文件即可。章节编号自动生成，交叉引用使用 `\label` / `\ref`。
- `main.tex`：字体、颜色、封面、标题、代码框和页面版式。
- `diagrams/diagram-01.tex` 至 `diagram-15.tex`：原生 TikZ 图，清单见 `diagrams/README.md`。
- `assets/`：正文引用的图片；`fonts/`：本报告使用的字体。
- `scripts/`：
  - `compile_xelatex.py`、`publish_pdf.py`：由 `.latexmkrc` 调用，分别负责编译前刷新高亮、编译后同步分发 PDF；
  - `highlight_yaml.py`：YAML 高亮配色与词法处理；
  - `plot_fitness_landscape.py`：生成 `assets/fitness_landscape.png`（`python3 scripts/plot_fitness_landscape.py assets/fitness_landscape.png`）。
- `old/`：原 Markdown 稿、旧 PPT、早期配图，以及已停用的 Markdown 转换、旧构建脚本和基于 Markdown 的核对工具，仅作存档。

`body-highlighted.tex` 和 `yaml-style.tex` 是自动生成文件，不要手工修改；删除后 `latexmk` 会重新生成。高亮使用 Pygments，逐字符核对代码内容，不需要 LaTeX shell-escape。行内标识使用 Alibaba Sans 并继承所在位置的字号/字重；代码块使用 DejaVu Sans Mono。

## 本机环境与依赖

本次安装的便携 TeX Live 位于：

```text
~/.local/share/kernel-schema-texlive
```

`~/.local/bin/latexmk` 和 `~/.local/bin/xelatex` 链接到该环境，未修改 shell 配置。若新终端提示命令不存在，可在当前终端执行：

```sh
export PATH="$HOME/.local/bin:$PATH"
```

`.latexmkrc` 自动补充本机便携 TeX 的工具路径。在其他机器上安装 TeX Live/MacTeX（含 XeLaTeX、latexmk）即可使用；此路径不存在时使用系统 PATH。Python 3 需要 `pygments`；重新生成适应性景观图还需要 `matplotlib` 与 `numpy`。

随附 DejaVu 字体许可在 `fonts/LICENSE_DEJAVU`。必要字体和图片已本地收齐，PDF 阅读者无需安装任何编译工具。
