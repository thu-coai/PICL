import sys

in_path = sys.argv[1]
out_path = sys.argv[2]
with open(in_path) as f:
    data = f.read()

docs = data.split("\n\n")

docs = [doc.strip() for doc in docs if len(doc.strip()) > 0]
docs = [doc.replace("\n", "<@x(x!>") for doc in docs]

with open(out_path, "w") as f:
    for doc in docs:
        f.write(doc + "\n")