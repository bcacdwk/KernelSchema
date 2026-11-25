# 基于 Schema 的 Kernel 表示与生成 · LaTeX

## 快捷编译

```sh
cd /Users/hanyong/KernelSchema/report/latex
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

- `body.tex`：正文与三个 YAML 代码块，直接修改此文件即可。
- `main.tex`：字体、颜色、封面、标题、代码框和页面版式。
- `diagrams/diagram-01.tex` 至 `diagram-13.tex`：原生 TikZ 流程图。
- `assets/`：七张原始图片；`fonts/`：本报告使用的字体。
- `scripts/highlight_yaml.py`：YAML 高亮配色与词法处理。

`body-highlighted.tex` 和 `yaml-style.tex` 是自动生成文件，不要手工修改；删除后 `latexmk` 会重新生成。高亮使用 Pygments，逐字符核对代码内容，不需要 LaTeX shell-escape。行内标识使用 Alibaba Sans 并继承所在位置的字号/字重；代码块使用 DejaVu Sans Mono。

## 从 Markdown 重新生成正文（可选）

```sh
python3 scripts/convert.py
latexmk
```

注意：转换会覆盖 `body.tex` 的手工修改，不改 TikZ。平时仅编辑 LaTeX 时，无需执行转换。转换器依赖上级 Markdown 及原图来源，是仓库内维护工具。

```sh
python3 scripts/convert.py --check  # 核对正文与当前 Markdown 的确定性转换是否一致
```

## 本机环境与依赖

本次安装的便携 TeX Live 位于：

```text
~/.local/share/kernel-schema-texlive
```

`~/.local/bin/latexmk` 和 `~/.local/bin/xelatex` 链接到该环境，未修改 shell 配置。若新终端提示命令不存在，可在当前终端执行：

```sh
export PATH="$HOME/.local/bin:$PATH"
```

`.latexmkrc` 自动补充本机便携 TeX 的工具路径。在其他机器上安装 TeX Live/MacTeX（含 XeLaTeX、latexmk）即可使用；此路径不存在时使用系统 PATH。Python 3 需要 `pygments`；只有 Markdown 转换额外需要 `markdown-it-py`。

```sh
python3 -m pip install pygments markdown-it-py
```

另保留 `scripts/build.py` 作为不使用 latexmk 的构建方式：

```sh
python3 scripts/build.py --engine xelatex
python3 scripts/publish_pdf.py
```

## 清理与验证资料

一次性 ZIP、逐图编译日志与快照、整页预览和重复测试输出已清理。保留 `qa/` 下轻量的核对脚本/清单，以及 `diagrams/_test/verify.py` 和验证摘要，供需要时复查。构建缓存、高亮生成文件等已列入本目录 `.gitignore`。

随附 DejaVu 字体许可在 `fonts/LICENSE_DEJAVU`。必要字体和图片已本地收齐，PDF 阅读者无需安装任何编译工具。
