import sys

conll_lines = []
inc_next = True
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith('-DOCSTART-'):
        inc_next = False
        continue
    else:
        if line or inc_next:
            conll_lines.append(line)
        inc_next = True
    

feats_lines = open(sys.argv[2]).readlines()
assert(len(conll_lines) == len(feats_lines))

for i in range(len(conll_lines)):
    infos = conll_lines[i].split()
    if not infos:
        print
        continue
    output = " ".join([infos[0], feats_lines[i].strip(), infos[-1]])
    print(output)
    