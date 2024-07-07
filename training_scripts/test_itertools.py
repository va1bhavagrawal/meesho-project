import itertools

a = [1, 2, 3]
b = [2, 3, 4]
c = itertools.chain(a, b)
print(f"{c = }")