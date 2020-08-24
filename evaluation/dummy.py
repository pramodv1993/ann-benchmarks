
def fun(k1,k2):
    print(k1, k2)

def callback(fun, args):
    print(args)
    fun(**args)

if __name__ == "__main__":
    callback(fun, dict(k1='v1',k2='v2'))
