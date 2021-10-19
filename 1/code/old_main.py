import na

import time


def CholeskySolve(U, c):
    HDagger = na.CholeskyDecDagger(U)
    y = na.LSolve(HDagger, c)
    # print(HDagger.T().conjugate()-HDagger.T())
    return na.USolve(HDagger.T().conjugate(), y)


n = 10
t1 = time.time()
print(na.GEM(na.Hilbert(n), na.vector([1 for _ in range(n)])))
t2 = time.time()
print(t2-t1)
# print(CholeskySolve(na.Hilbert(n), na.vector([1 for _ in range(n)])))
