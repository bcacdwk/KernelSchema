import json,pathlib,sys
W=pathlib.Path(__file__).parent;p=json.loads((W/'packs.json').read_text());u=json.loads((W/'indexed.json').read_text())
a,b=map(int,sys.argv[1:3]);full='full' in sys.argv
for x in p[a:b+1]:
 z=u[x['index']];o=z['occurrences'][0]
 print(f"#{x['index']} {z['family']} {o['node']}:{o['line']} {o['path']} [{len(z['occurrences'])} occurrences]")
 print(z['text'] if full else x['body']); print()
