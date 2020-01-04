class test1():
    a = None
    b = 'test1'

    def __init__(self):
        pass

class test2(test1):
    b = 'test2'

    def __init__(self):
        pass

t2 = test2()
print(t2.b)