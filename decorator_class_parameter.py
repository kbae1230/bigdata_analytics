class IsMultiple:

    def __init__(self,x):
        self.x = x
    
    def __call__(self, func):

        def wrapper(a, b):
            r = func(a, b)
            if r % self.x == 0:
                print('ok')
            else:
                print('nope')
            return r
        
        return wrapper

@IsMultiple(3)
def add(a, b):
    return a + b

print(add(10, 20))