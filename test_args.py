import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--a")
parser.add_argument("--b")
parser.add_argument("--c")


args, unknown = parser.parse_known_args()

print(unknown)

print(args.a)
print(args.b)
print(args.c)

