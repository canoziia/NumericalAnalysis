from numpy import arctanh, e, floor, log, pi
from scipy import integrate


def sumres(q):
    res = 0
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            for k in range(-R, R+1):
                n2 = i*i+j*j+k*k
                if n2 <= R*R:
                    res += e**(-(n2-q*q))/(n2-q*q)
    return res


def g(x, q, n2):
    return e**(x*q*q)*(pi/x)**1.5*e**(-pi**2*n2/x)


def intres(q):
    res = 0
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            for k in range(-R, R+1):
                n2 = i*i+j*j+k*k
                if n2 <= R*R and n2 != 0:
                    res += integrate.quad(g, 0, 1, args=(q, n2))[0]
    return res


R = 10
q = 0.5**0.5
# 常数
c = -5.053409213536382
print(sumres(q)+intres(q)+c)
