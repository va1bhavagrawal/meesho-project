def func(a, b):
    return a + b

args = {
    "a": 5,
    "b": 2,
    "c": 3,
}
print(func(**args))