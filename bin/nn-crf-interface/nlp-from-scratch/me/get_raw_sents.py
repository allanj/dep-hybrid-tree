import sys

sent = ""
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("-DOCSTART-"):
        continue
    if line == "":
        if sent != "":
            print sent.strip()
        sent = ""
        continue
    sent += line.split(" ")[0] + " "