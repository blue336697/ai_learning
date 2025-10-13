import this

message = "hello qiaoming.yuan"
message01 = " hello"
symbol = "\""
message02 = f"{symbol} {message} {symbol} {message01}"
print(message.title())
print(message.upper())
print(message.lower())
print(message + message01)
print(message01.strip())
print(message01.lstrip())
print(message01.rstrip())
print(message02)

# 3的3次方
num = 3 ** 3
print(num)

num01 = 3.0 ** 3
print(num01)

# 千分位 下划线
num02 = 1_000_000
print(num02)

fruit = ['apple', 'banana', 'orange']
print(fruit)
print(fruit[0])

fruit.append('pear')
fruit.insert(0, 'watermelon')
fruit.remove('orange')
del fruit[1]
someone = fruit.pop(1)
print(fruit)