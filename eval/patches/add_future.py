import os, sys, io
root = sys.argv[1]
changed = 0
for dirpath, _, files in os.walk(root):
    for fn in files:
        if not fn.endswith('.py'): continue
        p = os.path.join(dirpath, fn)
        with open(p, encoding='utf-8') as f: src = f.read()
        if 'from __future__ import annotations' in src: continue
        lines = src.split('\n')
        # find insertion point: after shebang and module docstring
        idx = 0
        if lines and lines[0].startswith('#!'): idx = 1
        # skip leading comments/blank
        # handle module docstring
        j = idx
        while j < len(lines) and (lines[j].strip()=='' or lines[j].lstrip().startswith('#')):
            j += 1
        if j < len(lines) and (lines[j].lstrip().startswith('"""') or lines[j].lstrip().startswith("'''")):
            q = lines[j].lstrip()[:3]
            # single-line docstring?
            if lines[j].strip().count(q) >= 2 and len(lines[j].strip())>3:
                idx = j+1
            else:
                k = j+1
                while k < len(lines) and q not in lines[k]: k += 1
                idx = k+1
        else:
            idx = idx  # insert at top (after shebang)
        lines.insert(idx, 'from __future__ import annotations')
        with open(p, 'w', encoding='utf-8') as f: f.write('\n'.join(lines))
        changed += 1
print(f'added future import to {changed} files')
