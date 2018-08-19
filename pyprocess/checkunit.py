unitfile = "test.txt"

dict = set()

with open(unitfile, 'r') as fin:
    for line in fin:
        line = line.rstrip()
        if line not in dict:
            dict.add(line)

for transition in dict:
    print(transition)

# print(dict)
print(len(dict))
