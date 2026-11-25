#!/usr/bin/env python3
"""Compile 13 original-scale TikZ pages; audit labels and physical extents."""
from pathlib import Path
import argparse, json, re, subprocess, sys, unicodedata
import fitz

HERE = Path(__file__).resolve().parent
DIAGRAMS = HERE.parent
ROOT = DIAGRAMS.parents[2]
SOURCE = ROOT / 'report/基于 Schema 的 Kernel 表示与生成.md'
PREAMBLE = r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage{fontspec}
\usepackage{xeCJK}
\setmainfont{Alibaba PuHuiTi}
\setsansfont{Alibaba PuHuiTi}
\setCJKmainfont{Alibaba PuHuiTi}
\setCJKsansfont{Alibaba PuHuiTi}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,calc,backgrounds,shapes.geometric}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\newsavebox{\diagramtestbox}
\begin{document}
'''

def normalize(s):
    s = unicodedata.normalize('NFC', s)
    return re.sub(r'\s+', '', s.replace('<br/>','').replace('<br>',''))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',default='tectonic',help='tectonic or a path to xelatex')
    parser.add_argument('--bundle',help='Optional tectonic bundle path')
    args=parser.parse_args()
    def command(tex, verbose=False):
        if 'tectonic' in Path(args.engine).name:
            cmd=[args.engine,'--keep-logs','--keep-intermediates']
            if verbose: cmd+=['--print']
            if args.bundle: cmd+=['--bundle',args.bundle]
            return cmd+[str(tex)]
        return [args.engine,'-interaction=nonstopmode','-halt-on-error','-output-directory='+str(HERE),str(tex)]
    files = [DIAGRAMS/f'diagram-{i:02d}.tex' for i in range(1,14)]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        sys.exit('Missing: '+', '.join(missing))
    body = []
    for i,p in enumerate(files,1):
        body.append(r'\sbox{\diagramtestbox}{\input{'+str(p)+'}}\n')
        body.append(r'\typeout{DIAGRAM-'+f'{i:02d}'+r' WIDTH=\the\wd\diagramtestbox HEIGHT=\the\dimexpr\ht\diagramtestbox+\dp\diagramtestbox\relax}'+'\n')
        body.append(r'\noindent\makebox[\linewidth][c]{\usebox{\diagramtestbox}}\par'+'\n')
        if i != 13: body.append('\\clearpage\n')
    tex = HERE/'all-diagrams.tex'
    tex.write_text(PREAMBLE+''.join(body)+'\\end{document}\n')
    proc = subprocess.run(command(tex,True),cwd=ROOT,capture_output=True,text=True)
    (HERE/'compile-output.txt').write_text(proc.stdout+'\n'+proc.stderr)
    if proc.returncode: sys.exit(proc.stdout+'\n'+proc.stderr)
    log=(HERE/'all-diagrams.log').read_text(errors='replace')
    dims={int(i):(float(w)*25.4/72.27,float(h)*25.4/72.27) for i,w,h in re.findall(r'DIAGRAM-(\d+) WIDTH=([\d.]+)pt\s*HEIGHT=([\d.]+)pt',log)}
    blocks=re.findall(r'```mermaid\s*\n(.*?)```',SOURCE.read_text(),re.S)
    doc=fitz.open(HERE/'all-diagrams.pdf')
    if len(doc)!=13: sys.exit(f'Expected 13 pages, got {len(doc)}')
    report=[]
    qa_path=ROOT/'report/latex/qa/diagram-labels.json'
    qa_labels={row['diagram']:row['labels'] for row in json.loads(qa_path.read_text())} if qa_path.exists() else {}
    for i,(page,block) in enumerate(zip(doc,blocks),1):
        # Every quoted Mermaid string is a node, group title, or edge label.
        expected=re.findall(r'"([^"]+)"',block)
        rendered=normalize(page.get_text(sort=False))
        (HERE/f'page-{i:02d}.txt').write_text(page.get_text())
        absent=[s for s in expected if normalize(s) not in rendered]
        w,h=dims[i]
        qa_absent=[label for label in qa_labels.get(i,[]) if normalize(label) not in rendered]
        row={'diagram':f'diagram-{i:02d}.tex','width_mm':round(w,2),'height_mm':round(h,2),'source_label_count':len(expected),'missing_labels':absent,'size_ok':w<=168.05 and h<=180.05}
        row['qa_missing_labels'] = qa_absent
        row['qa_label_count'] = len(qa_labels.get(i,[]))
        row['raster_images'] = len(page.get_images())
        spans = [span for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for span in line['spans'] if span['text'].strip()]
        row['min_font_size_bp'] = round(min(span['size'] for span in spans), 3)
        row['fonts'] = sorted({span['font'] for span in spans})
        report.append(row)
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),clip=page.rect).save(HERE/f'page-{i:02d}.png')
        print(f'{i:02d}: {w:.2f} x {h:.2f} mm; labels {len(expected)-len(absent)}/{len(expected)}; size={row["size_ok"]}')
        for text in absent: print('  MISSING:',text)
    (HERE/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    warnings=[line for line in log.splitlines() if any(s in line for s in ['Overfull','Missing character','Undefined control','LaTeX Warning','Package fontspec Warning'])]
    (HERE/'warnings.txt').write_text('\n'.join(warnings)+'\n')
    for i,p in enumerate(files,1):
        single=HERE/f'diagram-{i:02d}.tex'
        w,h=dims[i]
        single_preamble=PREAMBLE.replace(r'\usepackage[margin=20mm]{geometry}', r'\usepackage[paperwidth='+f'{w+8:.3f}mm,paperheight={h+10:.3f}mm,margin=4mm'+r']{geometry}')
        single.write_text(single_preamble+r'\sbox{\diagramtestbox}{\input{'+str(p)+r'}}\noindent\raisebox{\dp\diagramtestbox}{\usebox{\diagramtestbox}}'+'\n'+r'\end{document}'+'\n')
        result=subprocess.run(command(single),cwd=ROOT,capture_output=True,text=True)
        if result.returncode: sys.exit(result.stdout+'\n'+result.stderr)
    for i in range(1,14):
        single_doc=fitz.open(HERE/f'diagram-{i:02d}.pdf')
        if len(single_doc)!=1: sys.exit(f'diagram-{i:02d}.pdf: expected one page, got {len(single_doc)}')
        report[i-1]['standalone_pages']=len(single_doc)
        report[i-1]['engine']=args.engine
    (HERE/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Standalone PDFs: 13 compiled, each exactly one page.')
    print('Warnings:',len(warnings))
    if warnings: print('\n'.join(warnings))
    if any(r['missing_labels'] or r['qa_missing_labels'] or not r['size_ok'] or r['raster_images'] for r in report): sys.exit(1)

if __name__=='__main__': main()
