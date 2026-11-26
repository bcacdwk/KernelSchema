#!/usr/bin/env python3
"""Check rendered Chinese text against source; render page contact sheets for review."""
import argparse, json, re, hashlib
from pathlib import Path
import fitz
from PIL import Image, ImageDraw
p=argparse.ArgumentParser(); p.add_argument('pdf',type=Path); a=p.parse_args()
root=Path(__file__).parent
inventory=json.loads((root/'source-inventory.json').read_text())
doc=fitz.open(a.pdf)
full=''.join(page.get_text(clip=fitz.Rect(0, 48, page.rect.width, page.rect.height-52)) for page in doc)
flat=re.sub(r'\s+','',full)
missing=[]; checked=0
for block in inventory['blocks']:
 text=block.get('text',block.get('code',''))
 for segment in re.findall(r'[\u4e00-\u9fff]{6,}',text):
  checked+=1
  if segment not in flat: missing.append({'lines':block['lines'],'segment':segment})
report={'pages':len(doc),'source_sha256':inventory['sha256'],'chinese_segments_checked':checked,'missing_segments':missing}
(root/'text-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
for start in range(0,len(doc),12):
 thumbs=[]
 for i in range(start,min(start+12,len(doc))):
  pix=doc[i].get_pixmap(matrix=fitz.Matrix(.42,.42),alpha=False)
  im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
  tile=Image.new('RGB',(270,390),'#e7e9ed'); tile.paste(im,((270-im.width)//2,20));ImageDraw.Draw(tile).text((10,3),str(i+1),fill='black'); thumbs.append(tile)
 sheet=Image.new('RGB',(1080,390*((len(thumbs)+3)//4)),'white')
 for j,im in enumerate(thumbs):sheet.paste(im,((j%4)*270,(j//4)*390))
 sheet.save(root/f'pages-{start+1:02d}.png')
