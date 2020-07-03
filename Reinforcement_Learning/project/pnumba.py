import numpy as np
from numba import jit, autojit, prange

@jit(nopython=True, parallel=True)
def gen(N):
    arr = np.ones(N)
    for i in prange(N):
        arr[i] = np.random.randint(-100, 101)
    return arr

@jit
def is_sorted(arr):
    for i in range(len(arr)-1):
        if arr[i] > arr[i+1]:
            return False
    return True

@jit(nopython=True, parallel=True)
def expectation(arr, maxiter, eps):
    length = arr.size
    old_exp = 0
    exp = 0
    convg = 0
    for n in prange(1, maxiter+1):
        a = arr.copy()
        count = 0
        while not is_sorted(a):
            i = np.random.randint(length)
            j = np.random.randint(length)
            while i==j:
                j = np.random.randint(length)
            temp = a[i]
            a[i] = a[j]
            a[j] = temp
            count += 1
            
        exp = exp + (count - exp)/n
        if abs(exp-old_exp) < eps:
            convg += 1
            if convg > 10: return exp
        else:
            convg = 0
        old_exp = exp
    else:
        return -exp

    
z = 1000000000
while True:
    print(len(gen(z)))
    z+=10000000
    a = np.array([1,5,9,7,1,6,3,4])
    print(expectation(a, 1000000000, 0.000001))
    if z > 1.3e8:
        break