import re, sys
src = open(r'D:\GitHub\TinyLM\tinyLmPaper.latex', encoding='utf-8').read()
errors = []

# environment balance
begins = re.findall(r'\\begin\{([^}]+)\}', src)
ends = re.findall(r'\\end\{([^}]+)\}', src)
from collections import Counter
cb, ce = Counter(begins), Counter(ends)
if cb != ce:
    errors.append(f'env mismatch: begins-ends={cb-ce}, ends-begins={ce-cb}')

# brace balance (strip escaped backslashes and escaped braces first)
tmp = src.replace('\\\\', '  ').replace('\\{', '  ').replace('\\}', '  ')
depth = 0
for i, ch in enumerate(tmp):
    if ch == '{':
        depth += 1
    elif ch == '}':
        depth -= 1
    if depth < 0:
        errors.append(f'brace went negative at char {i}')
        break
if depth != 0:
    errors.append(f'final brace depth {depth}')

# odd number of $ per line (naive math toggle check)
for n, line in enumerate(src.splitlines(), 1):
    stripped = line.replace('\\$', '')
    if stripped.count('$') % 2:
        errors.append(f'odd $ count line {n}: {line.strip()[:80]}')

# figure files exist
import os
for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', src):
    p = os.path.join(r'D:\GitHub\TinyLM', m.group(1))
    if not os.path.exists(p):
        errors.append(f'missing figure: {m.group(1)}')

# labels and refs
labels = set(re.findall(r'\\label\{([^}]+)\}', src))
refs = set(re.findall(r'\\[a-zA-Z]*ref\{([^}]+)\}', src))
missing = refs - labels
if missing:
    errors.append(f'refs without labels: {missing}')

# cites vs bibitems
cites = set()
for m in re.finditer(r'\\cite\{([^}]+)\}', src):
    cites.update(k.strip() for k in m.group(1).split(','))
bibs = set(re.findall(r'\\bibitem\{([^}]+)\}', src))
if cites - bibs:
    errors.append(f'cites without bibitem: {cites - bibs}')
if bibs - cites:
    errors.append(f'uncited bibitems: {bibs - cites}')

# leftover first person outside bibliography
body = src.split('\\begin{thebibliography}')[0]
fp = re.findall(r'\b(I|my|mine|myself)\b', body)
if fp:
    errors.append(f'first-person leftovers: {fp}')

print('ERRORS:' if errors else 'ALL CHECKS PASSED')
for e in errors:
    print(' -', e)
sys.exit(1 if errors else 0)
