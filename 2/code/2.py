import na
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi
from scipy.interpolate import CubicSpline


def f(x):
    return 1/(1+25*x**2)


X = np.linspace(-1, 1, 21)
Y = np.array([f(X[i]) for i in range(len(X))])\

P20 = na.Neville(X, Y)
fT = na.ChebyshevExpansion(f, 20)
fC = na.CubicSpline(X, Y, out="func")
# fpp = CubicSpline(X, Y, bc_type="natural")


def view(f1, filename, vxp):
    vx = np.linspace(-1, 1, 1001)
    vy1 = np.array([f(vx[i]) for i in range(len(vx))])
    vy2 = np.array([f1(vx[i]) for i in range(len(vx))])
    plt.plot(vx, vy1)
    plt.plot(vx, vy2)
    plt.savefig(filename+".jpg")
    plt.clf()
    vy = np.array([abs(f1(vx[i])-f(vx[i])) for i in range(len(vx))])
    plt.plot(vx, vy)
    vyp = np.array([abs(f1(vxp[i])-f(vxp[i])) for i in range(len(vxp))])
    plt.scatter(vxp, vyp)
    plt.savefig(filename+".sub.jpg")
    plt.clf()


def latex():
    res = ""
    xl = np.linspace(-1, 1, 41)
    xcl = []
    for k in range(20):
        xcl.append(cos(pi*(k+0.5)/20))
        xcl.append(cos(pi*(k+0.5)/20)/2+cos(pi*(k+1.5)/20)/2)
    xcl.pop()
    xcl.extend([0, 0])  # 到时候去掉这两个

    for i in range(41):
        res += "%d&%f&%f&%f&%f&%f&%f&%f\\\\\n" % (i, f(xl[i]), P20(xl[i]), fT(
            xcl[i]), fC(xcl[i]), abs(P20(xl[i])-f(xl[i])), abs(fT(xcl[i])-f(xcl[i])), abs(fC(xl[i])-f(xl[i])))
    print(res)


view(P20, "P20", np.linspace(-1, 1, 41))
xcl = []
for k in range(20):
    xcl.append(cos(pi*(k+0.5)/20))
    xcl.append(cos(pi*(k+0.5)/20)/2+cos(pi*(k+1.5)/20)/2)
view(fT, "fChebyshev", xcl)
view(fC, "fCubicSpline", np.linspace(-1, 1, 41))

latex()
