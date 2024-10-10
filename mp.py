from multiprocessing import Pool

def pool_func(a):
    print(a)

with Pool(3) as p:
    mem = list(range(10000))
    p.map(pool_func, list(range(9))) 
