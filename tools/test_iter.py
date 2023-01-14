class A:
    def __init__(self) -> None:
        self.epoch = 0
        self.x = list(range(10))
        
    def __iter__(self):
        self.epoch += 1
        return self.x.__iter__()
    
    
a = A()

for x in a:
    print(x)
    
print(a.epoch)

for x in a:
    print(x)

print(a.epoch)
