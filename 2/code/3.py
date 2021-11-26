from numpy import cos, pi, sin
import numpy as np
import na
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def xt(phi):
    return (1-cos(phi))*cos(phi)


def yt(phi):
    return (1-cos(phi))*sin(phi)


n = 8
phi = [2*pi*i/n for i in range(n+1)]
xtList = [xt(t) for t in phi]
ytList = [yt(t) for t in phi]


def latex():
    res = ""
    for i in xtList:
        res += "%f&" % i
    res = res[:-1]+r"\\"+"\n"
    for i in ytList:
        res += "%f&" % i
    res = res[:-1]+r"\\"+"\n"
    print(res)


latex()


Sx = na.CubicSpline(phi, xtList, case=3, out="func")
Sy = na.CubicSpline(phi, ytList, case=3, out="func")
# Sxfk = CubicSpline(phi, xtList, bc_type="periodic")
# xxx = np.linspace(2*pi-0.5, 2*pi, 10)
# yyy = [Sx(xx) for xx in xxx]
# rrr = ""
# for iii in range(len(xxx)):
#     rrr += "{%f,%f}," % (xxx[iii], yyy[iii])
# print(rrr)

xmat = na.CubicSpline(phi, xtList, case=3, out="mat")
ymat = na.CubicSpline(phi, ytList, case=3, out="mat")


def f(x):
    return Sx(x)-(xmat[0][0]+xmat[0][1]*x+xmat[0][2]*x**2+xmat[0][3]*x**3)


def latex2():
    res = ""
    for i in xmat:
        for j in i:
            res += "&%.3f" % j
        res += r"\\"+"\n"
    res += "\n"
    for i in ymat:
        for j in i:
            res += "&%.3f" % j
        res += r"\\"+"\n"
    res += "\n"
    print(res)


latex2()


def view(f1, filename):
    plt.clf()
    vx = np.linspace(0, 2*pi, 1001)
    vy = np.array([f1(x) for x in vx])
    plt.plot(vx, vy)
    plt.savefig(filename+".jpg")
    plt.clf()


# view(Sx, "Sx")


def cardioid():
    plt.clf()
    vt = np.linspace(0, 2*pi, 1001)
    vx = np.array([xt(t) for t in vt])
    vy = np.array([yt(t) for t in vt])
    # 内插
    vSx = np.array([Sx(t) for t in vt])
    vSy = np.array([Sy(t) for t in vt])
    plt.plot(vx, vy)
    plt.plot(vSx, vSy)
    plt.scatter(xtList, ytList)
    plt.savefig("cardioid.jpg")
    plt.clf()


cardioid()
