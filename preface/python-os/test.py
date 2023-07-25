variables = {}
x = 'name = "John"\nprint(name)'
exec(x, variables)  # 输出 John
y = """
def main():
    sys_write('Hello, OS World')
"""
exec(y, variables)
print(variables)

# John
# {'__builtins__': {'__name__': 'builtins', ..., 'name': 'John', 'main': <function main at 0x7f8f8ad6bd90>}