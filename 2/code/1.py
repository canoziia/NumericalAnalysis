import na
import numpy as np
from numpy import pi, sin
import matplotlib.pyplot as plt

n = 9
f = na.Neville([pi/2/(n-1)*i for i in range(n)],
               [sin(pi/2/(n-1)*i) for i in range(n)])


x = np.linspace(0, pi/2, 1001)


# def g(x):
#     return [(pi/2)**4/5/4/3/2/1]*len(x)


plt.plot(x, f(x)-sin(x))
# plt.plot(x, sin(x))
plt.savefig("1.jpg")


def findN():
    n = 1
    s = pi/2/2
    while True:
        if s < 1e-8:
            return n
        n += 1
        s *= pi/2/(n+1)


# print(findN())
