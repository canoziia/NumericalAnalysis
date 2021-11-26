from numpy import cos, e, sqrt, tan, pi
import na
from scipy.special import roots_hermite


def f(x):
    return e**(-x*x)*cos(x)


def latexLadder():
    nl = [1, 5, 10, 15]
    cutl = [100, 1000, 10000, 100000]
    res = ""
    for n in nl:
        for cut in cutl:
            res += "%s&" % na.Integrate(f, -n, n, method="ladder")
        res = res[:-1]+r"\\"+"\n"
    print(res)


def latexExtra():
    nl = [1, 5, 10, 15]
    ml = [14, 15, 16, 17]
    res = ""
    for n in nl:
        for m in ml:
            res += "%s&" % na.Integrate(f, -n, n,
                                        method="extrapolation", exponent=m)
        res = res[:-1]+r"\\"+"\n"
    print(res)


latexLadder()
latexExtra()


def Gauss(f, n):
    xl, yl = roots_hermite(n)
    res = 0
    for i in range(len(xl)):
        res += f(xl[i])*yl[i]
    return res


for n in [5, 10, 15, 20]:
    print(Gauss(cos, n))
print(sqrt(pi)*e**-0.25)
