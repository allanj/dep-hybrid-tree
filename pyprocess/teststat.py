geoTestFile = "../data/org-geo/embedding-train-N600-en.err"

f = open(geoTestFile, 'r')
## prevLine = None
line = f.readline()
sents = []
tf = []
while line:
    line = line.rstrip()
    #print(line)
    if line == "=INPUT=":
        nl = f.readline().rstrip()
        sents.append(nl)
        tf.append(prevLine.split(":\t")[1])
    prevLine = line
    line = f.readline()
f.close()
#print(sents)
#print(tf)

qword = {} ##dictionary to store
wrongQWord = {}

for i in range(len(sents)):
    print(sents[i] + " " + tf[i])
    sent = sents[i]
    words = sent.split(" ")
    if words[0] in qword:
        qword[words[0]] = qword[words[0]] + 1
    else:
        qword[words[0]] = 1
    if tf[i] == "false":
        if words[0] in qword:
            wrongQWord[words[0]] = wrongQWord[words[0]] + 1
        else:
            wrongQWord[words[0]] = 1

print(qword)