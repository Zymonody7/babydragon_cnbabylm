import re, sys, io

def convert(src):
    lines = src.split('\n')
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = re.match(r'^(\s*)match\s+(.+?):\s*$', line)
        if not m:
            out.append(line); i += 1; continue
        indent = m.group(1); expr = m.group(2).strip()
        i += 1
        case_indent = None
        first = True
        # process case blocks
        while i < n:
            cl = lines[i]
            if cl.strip() == '' or cl.lstrip().startswith('#'):
                out.append(cl); i += 1; continue
            cm = re.match(r'^(\s*)case\s+(.+?):\s*$', cl)
            if not cm:
                break
            ci = cm.group(1)
            if case_indent is None: case_indent = ci
            if len(ci) <= len(indent):  # dedented past match → end
                break
            pat = cm.group(2).strip()
            if pat == '_':
                out.append(f'{indent}else:')
            else:
                # split on | for OR patterns
                parts = [p.strip() for p in pat.split('|')]
                if len(parts) == 1:
                    cond = f'{expr} == {parts[0]}'
                else:
                    cond = f'{expr} in ({", ".join(parts)})'
                kw = 'if' if first else 'elif'
                out.append(f'{indent}{kw} {cond}:')
            first = False
            i += 1
            # body: lines more indented than case_indent, dedent by (len(case_indent)-len(indent))
            dedent = len(case_indent) - len(indent)
            while i < n:
                bl = lines[i]
                if bl.strip() == '':
                    out.append(bl); i += 1; continue
                bind = len(bl) - len(bl.lstrip())
                if bind <= len(case_indent):
                    break
                out.append(bl[dedent:]); i += 1
    return '\n'.join(out)

for path in sys.argv[1:]:
    with open(path, encoding='utf-8') as f: src = f.read()
    new = convert(src)
    with open(path+'.bak', 'w', encoding='utf-8') as f: f.write(src)
    with open(path, 'w', encoding='utf-8') as f: f.write(new)
    # syntax check (compile under current python)
    try:
        compile(new, path, 'exec'); print(f'OK {path}')
    except SyntaxError as e:
        print(f'SYNTAXERR {path}: {e}')
