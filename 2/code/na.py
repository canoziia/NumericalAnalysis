from typing import *
import copy
from numpy import cos, pi


class LengthMisMatch(Exception):
    pass


class UnknownType(Exception):
    pass


class UndefinedValue(Exception):  # 超出定义域
    pass


class tensor(list):
    ...


Num = TypeVar("Num")
Elem = Union[Num, tensor]


def conjugate(x: Elem) -> Elem:
    if hasattr(x, "conjugate"):
        return x.conjugate()
    else:
        return x


def isNum(x) -> bool:
    return not hasattr(x, "__len__")


class tensor(list):
    def __init__(self, arr: Iterable) -> None:
        list.__init__([])
        self.__import__(arr)

    def __import__(self, arr: Iterable) -> None:
        if len(arr) == 0 or isNum(arr[0]):
            # for x in arr:
            #     self.append(x)
            # return
            self.extend(arr)
            return
        for x in arr:
            self.append(self.__class__(x))

    def shape(self) -> List:
        if len(self) == 0:
            return [0]
        elif isNum(self[0]):
            return [len(self)]
        res = self[0].shape()
        res.insert(0, len(self))
        return res

    def conjugate(self):
        res = self.__class__([])
        for i in range(len(self)):
            res.append(conjugate(self[i]))
        return res

    def Conjugate(self) -> None:
        for i in range(len(self)):
            self[i] = conjugate(self[i])

    def chkShape(self, x) -> None:
        if self.shape() != x.shape():
            raise LengthMisMatch

    def getitem(self, i: List):  # 返回一个数或一个tensor
        if len(i) == 0:
            return self
        if len(i) > 1:
            return self[i[0]].getitem(i[1:])
        return self[i[0]]

    def setitem(self, i: List, o):
        if len(i) > 1:
            self[i[0]].setitem(i[1:], o)
            return
        self[i[0]] = o
        return

    def traverse(self, pos: List, f):
        sshape = self.shape()
        if len(pos) == len(sshape):
            # self.setitem(pos, f(pos))
            f(self, pos)
            return
        for i in range(sshape[len(pos)]):
            pos.append(i)
            self.traverse(pos, f)
            pos.pop()

    def changeRow(self, d: int, r1: int, r2: int):
        if r1 == r2:
            return

        def f(s, l):
            if l[d] != r1:
                return
            t1 = s.getitem(l)
            l[d] = r2
            t2 = s.getitem(l)
            s.setitem(l, t1)
            l[d] = r1
            s.setitem(l, t2)

        self.traverse([], f)

    def T(self):
        if len(self.shape()) != 2:
            return
        res = self.__class__([])
        # oldrow
        orow = self.shape()[0]
        if orow == 0:
            return res
        for nrow in range(self.shape()[1]):
            res.append(self.__class__([]))
            for ncolomn in range(orow):
                res[nrow].append(self[ncolomn][nrow])
        return res

    def __add__(self, x: Elem):
        if isNum(x):
            return self.__class__([(self[i]+x) for i in range(len(self))])
        self.chkShape(x)
        return self.__class__([(self[i]+x[i]) for i in range(len(self))])

    def __radd__(self, x: Elem):
        return self.__add__(x)

    def __pos__(self):
        return self.__class__([(+self[i]) for i in range(len(self))])

    def __iadd__(self, x: Elem):
        return self.__add__(x)

    def __sub__(self, x: Elem):
        if isNum(x):
            return self.__class__([(self[i]-x) for i in range(len(self))])
        self.chkShape(x)
        return self.__class__([(self[i]-x[i]) for i in range(len(self))])

    def __rsub__(self, x: Elem):
        if isNum(x):
            return self.__class__([(x-self[i]) for i in range(len(self))])
        self.chkShape(x)
        return self.__class__([(x[i]-self[i]) for i in range(len(self))])

    def __neg__(self):
        return self.__class__([(-self[i]) for i in range(len(self))])

    def __truediv__(self, x: Elem):
        if isNum(x):
            return self.__class__([(self[i]/x) for i in range(len(self))])
        self.chkShape(x)
        return self.__class__([(self[i]/x[i]) for i in range(len(self))])

    def __rtruediv__(self, x: Elem):
        if isNum(x):
            return self.__class__([(x/self[i]) for i in range(len(self))])
        self.chkShape(x)
        return self.__class__([(x[i]/self[i]) for i in range(len(self))])

    # 标积

    def __mul__(self, x: Elem):
        if isNum(x):
            res = self.__class__([])
            for i in self:
                res.append(i*x)
            return res
        sshape = self.shape()
        xshape = x.shape()
        suml = sshape[-1]
        if sshape[-1] != xshape[0]:
            raise LengthMisMatch
        sshape.pop()
        xshape.pop(0)
        resShape = sshape+xshape
        res = zeros(resShape)

        def walk(sl, xl):
            nonlocal res
            if len(sl) == len(sshape) and len(xl) == len(xshape):
                s = 0
                if len(x.shape()) == 1:
                    for i in range(suml):
                        s += self.getitem(sl)[i]*x[i]
                else:
                    for i in range(suml):
                        s += self.getitem(sl)[i]*x[i].getitem(xl)
                if isNum(res):
                    res = s
                    return
                res.setitem(sl+xl, s)
            elif len(sl) < len(sshape):
                for i in range(sshape[len(sl)]):
                    sl.append(i)
                    walk(sl, xl)
                    sl.pop()
            elif len(sl) == len(sshape) and len(xl) < len(xshape):
                for i in range(xshape[len(xl)]):
                    xl.append(i)
                    walk(sl, xl)
                    xl.pop()
        walk([], [])
        return res

    def __rmul__(self, x: Elem):
        return self.__mul__(x)


def zeros(shape: List):
    if len(shape) == 0:
        return 0
    elif len(shape) == 1:
        return tensor([0]*shape[0])
    res = tensor([])
    tmpshape = copy.deepcopy(shape)
    tmpshape.pop(0)
    child = zeros(tmpshape)
    for i in range(shape[0]):
        res.append(copy.deepcopy(child))
    return res


def I(n: int):
    if n == 0:
        return 1
    else:
        res = zeros([n, n])
        for i in range(n):
            res[i][i] = 1
    return res


def PmtMat(n: int, r1: int, r2: int):
    res = zeros([n, n])
    for i in range(n):
        res[i][i] = 1
    res[r1][r1] = 0
    res[r2][r2] = 0
    res[r1][r2] = 1
    res[r2][r1] = 1
    return res


def Frobinius(n: int, i: int, l: List):  # i 主元所在行，n 矩阵维，l ari/aii
    res = zeros([n, n])
    for t in range(n):
        res[t][t] = 1
    for t in range(i+1, n):
        res[t][i] = -l[t-i-1]
    return res


# U*res=c

def Hilbert(n: int):
    res = zeros([n, n])
    for i in range(n):
        for j in range(n):
            res[i][j] = 1/(i+j+1)
    return res


def USolve(U: tensor, c: tensor) -> tensor:
    n = len(U)
    res = tensor([0] * n)
    for i in range(n-1, -1, -1):
        t = 0
        for k in range(i+1, n):
            t += U[i][k]*res[k]
        res[i] = (c[i]-t)/U[i][i]
    return res


def LSolve(L: tensor, c: tensor) -> tensor:
    n = len(L)
    res = tensor([0]*n)
    for i in range(n):
        t = 0
        for k in range(i):
            t += L[i][k]*res[k]
        res[i] = (c[i]-t)/L[i][i]
    return res


def GEM(U: tensor, c: tensor) -> tensor:
    row = len(U)
    i = 0
    for i in range(row):
        for it in range(i, row):
            if U[it][i] != 0:
                U.changeRow(0, it, i)
                c.changeRow(0, it, i)
        l = [U[t][i]/U[i][i] for t in range(i+1, row)]
        f = Frobinius(row, i, l)
        U = f*U
        c = f*c
    # res = tensor([0] * row)
    # for i in range(row-1, -1, -1):
    #     t = 0
    #     for k in range(i+1, row):
    #         t += U[i][k]*res[k]
    #     res[i] = (U[i][col-1]-t)/U[i][i]
    ctmp = tensor([])
    for i in range(row):
        ctmp.append(c[i])
    return USolve(U, ctmp)


def CholeskyDecDagger(U: tensor) -> tensor:
    if len(U) == 0 or len(U) != len(U[0]):
        raise LengthMisMatch
    n = len(U)
    H = zeros([n, n])
    H[0][0] = (U[0][0])**0.5
    for i in range(1, n):
        for j in range(i):
            t = 0
            for k in range(j):
                t += H[i][k]*H[j][k]
            H[i][j] = (U[i][j]-t)/H[j][j]
        t = 0
        for k in range(i):
            t += H[i][k]*H[i][k]
        H[i][i] = (U[i][i]-t)**0.5
    return H


def CholeskyDec(U: tensor) -> tensor:
    return CholeskyDecDagger(U).T().conjugate()


def CholeskySolve(U: tensor, c: tensor):
    HDagger = CholeskyDecDagger(U)
    y = LSolve(HDagger, c)
    return USolve(HDagger.T(), y)

#T[j][k-1]+(T[j][k-1]-T[j-1][k-1]) / ((x-points[j-k][0])/(x-points[j][0])-1)


def Neville(dataX: List[Num], dataY: List[Num]):   # List[tuple]
    def res(x):
        n = len(dataX)-1
        T = [[dataY[i]] for i in range(n+1)]
        for j in range(1, n+1):
            for k in range(1, j+1):
                T[j].append((T[j][k-1]*(x-dataX[j-k])-T[j-1][k-1]
                            * (x-dataX[j]))/(dataX[j]-dataX[j-k]))
        return T[n][n]
    return res


def TridiagonalSolve(A: tensor, b: tensor):  # Ax=b
    n = A.shape()[0]
    L = I(n)
    U = zeros([n, n])
    U[0][0] = A[0][0]
    for i in range(1, n):
        L[i][i-1] = A[i][i-1]/U[i-1][i-1]
        U[i][i] = A[i][i]-L[i][i-1]*A[i-1][i]
        U[i-1][i] = A[i-1][i]
    y = LSolve(L, b)
    return USolve(U, y)


def CubicSpline(dataX: List, dataY: List, case: int = 1, D: tuple = (0, 0), out="func"):
    n = len(dataX)-1
    h = tensor([0])  # 从0-n，0没用
    for i in range(n):
        h.append(dataX[i+1]-dataX[i])
    A = zeros([n+1, n+1])
    b = zeros([n+1])
    if case == 1:  # M_0=M_n=0,自然边界
        A[0][0] = 1
        # b[0]=0
        A[n][n] = 1
        # b[n]=0
    elif case == 2:  # 边界导数相等
        A[0][0] = -h[1]/3
        A[0][1] = -h[1]/6
        b[0] = D[0]-(dataY[1]-dataY[0])/h[1]
        A[n][n] = h[n]/3
        A[n][n-1] = h[n]/6
        b[n] = D[1]-(dataY[n]-dataY[n-1])/h[n]
    elif case == 3:  # 周期边界
        # 此时y[n]=y[0]，矩阵不是三对角
        A[0][n-1] = h[n]/6
        A[0][0] = (h[n]+h[1])/3
        A[0][1] = h[1]/6
        b[0] = (dataY[1]-dataY[0])/h[1]-(dataY[0]-dataY[n-1])/h[n]
        A[n][n-1] = h[n]/6
        A[n][n] = (h[n]+h[1])/3
        A[n][1] = h[1]/6
        b[n] = (dataY[1]-dataY[n])/h[1]-(dataY[n]-dataY[n-1])/h[n]
    for i in range(1, n):
        A[i][i-1] = h[i]/6
        A[i][i] = (h[i]+h[i+1])/3
        A[i][i+1] = h[i+1]/6
        b[i] = (dataY[i+1]-dataY[i])/h[i+1]-(dataY[i]-dataY[i-1])/h[i]
    if case == 1 or case == 2:
        M = TridiagonalSolve(A, b)
    else:
        M = GEM(A, b)

    if out == "mat":
        resmat = []
        for j in range(n):
            resmat.append([
                dataY[j]+dataX[j]*(dataY[j]-dataY[j+1])/h[j+1]++dataX[j]*(M[j+1]-M[j])*h[j+1]/6+(
                    M[j]*dataX[j+1]**3-M[j+1]*dataX[j]**3)/h[j+1]/6 - h[j+1]**2*M[j]/6,
                (dataY[j+1]-dataY[j])/h[j+1]+h[j+1]*(M[j]-M[j+1]) /
                6+(M[j+1]*dataX[j]**2-M[j]*dataX[j+1]**2)/h[j+1]/2,
                (M[j]*dataX[j+1]-M[j+1]*dataX[j])/h[j+1]/2,
                (M[j+1]-M[j])/h[j+1]/6
            ])
        return resmat

    def res(x):
        for i in range(n):
            if dataX[i] <= x and x <= dataX[i+1]:
                # a = (dataY[i+1]-dataY[i])/h[i+1]-h[i+1]*(M[i+1]-M[i])/6
                # b = dataY[i]-M[i]*h[i+1]**2/6
                # s = M[i]*(dataX[i+1]-x)**3/h[i+1]/6+M[i+1] * \
                #     (x-dataX[i])**3/h[i+1]/6+a*(x-dataX[i])+b
                s = M[i]*(dataX[i+1]-x)**3/h[i+1]/6+M[i+1]*(x-dataX[i])**3/h[i+1]/6+(x-dataX[i])*(
                    (dataY[i+1]-dataY[i])/h[i+1]-h[i+1]*(M[i+1]-M[i])/6)+dataY[i]-M[i]*h[i+1]**2/6
                return s
    return res


def Chebyshev(n: int, x: Num):
    if n < 0 or type(n) != int:
        raise UndefinedValue
    # if n == 0:
    #     return 1
    # elif n == 1:
    #     return x
    else:
        c = [1, x]
        for i in range(2, n+1):
            c.append(2*x*c[i-1]-c[i-2])
        return c[n]


def ChebyshevExpansion(f: Callable[[Num], Num], n: int):
    c = []
    for m in range(n):
        t = 0
        for k in range(n):
            t += f(cos(pi*(k+0.5)/n))*cos(pi*m*(k+0.5)/n)
        c.append(2*t/n)

    def res(x):
        t = c[0]/2
        for k in range(1, n):
            t += c[k]*Chebyshev(k, x)
        return t
    return res


def Integrate(f: Callable[[Num], Num], inf: Num, sup: Num, method="ladder", cut: int = 1000, exponent: int = 10):
    if method == "ladder":
        res = 0
        h = (sup-inf)/cut
        for i in range(1, cut):
            res += f(inf+i*h)
        res += (f(inf)+f(sup))/2
        return res*h
    elif method == "extrapolation":
        hList = [(sup-inf)/2**i for i in range(exponent+1)]
        resList = [Integrate(f, inf, sup, method="ladder", cut=2**i)
                   for i in range(exponent+1)]
        return Neville(hList, resList)(0)
