from typing import *
import copy


class LengthMisMatch(Exception):
    pass


class UnknownType(Exception):
    pass


def conjugate(x):
    if hasattr(x, "conjugate"):
        return x.conjugate()
    else:
        return x


def isNum(x):
    return type(x) == int or type(x) == float or type(x) == complex


class tensor(list):
    def __init__(self, arr):
        list.__init__([])
        self.__import__(arr)

    def __import__(self, arr) -> None:
        if len(arr) == 0 or isNum(arr[0]):
            # for x in arr:
            #     self.append(x)
            # return
            self.extend(arr)
            return
        for x in arr:
            self.append(self.__class__(x))

    def shape(self):
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

    def chklen(self, x):
        if self.shape() != x.shape():
            raise LengthMisMatch

    def getitem(self, i):
        if len(i) == 0:
            return self
        if len(i) > 1:
            return self[i[0]].getitem(i[1:])
        return self[i[0]]

    def setitem(self, i, o):
        if len(i) > 1:
            self[i[0]].setitem(i[1:], o)
            return
        self[i[0]] = o
        return

    def traverse(self, pos, f):
        sshape = self.shape()
        if len(pos) == len(sshape):
            # self.setitem(pos, f(pos))
            f(self, pos)
            return
        for i in range(sshape[len(pos)]):
            pos.append(i)
            self.traverse(pos, f)
            pos.pop()

    def changeRow(self, d, r1, r2):
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

    def __add__(self, x):
        if isNum(x):
            return self.__class__([(self[i]+x) for i in range(len(self))])
        self.chklen(x)
        return self.__class__([(self[i]+x[i]) for i in range(len(self))])

    def __radd__(self, x):
        return self.__add__(x)

    def __pos__(self):
        return self.__class__([(+self[i]) for i in range(len(self))])

    def __iadd__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        if isNum(x):
            return self.__class__([(self[i]-x) for i in range(len(self))])
        self.chklen(x)
        return self.__class__([(self[i]-x[i]) for i in range(len(self))])

    def __rsub__(self, x):
        if isNum(x):
            return self.__class__([(x-self[i]) for i in range(len(self))])
        self.chklen(x)
        return self.__class__([(x[i]-self[i]) for i in range(len(self))])

    def __neg__(self):
        return self.__class__([(-self[i]) for i in range(len(self))])

    def __truediv__(self, x):
        if isNum(x):
            return self.__class__([(self[i]/x) for i in range(len(self))])
        self.chklen(x)
        return self.__class__([(self[i]/x[i]) for i in range(len(self))])

    def __rtruediv__(self, x):
        if isNum(x):
            return self.__class__([(x/self[i]) for i in range(len(self))])
        self.chklen(x)
        return self.__class__([(x[i]/self[i]) for i in range(len(self))])

    # 标积

    def __mul__(self, x):
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

    def __rmul__(self, x):
        return self.__mul__(x)


def zeros(shape: list):
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


def PmtMat(n, r1, r2):
    res = zeros([n, n])
    for i in range(n):
        res[i][i] = 1
    res[r1][r1] = 0
    res[r2][r2] = 0
    res[r1][r2] = 1
    res[r2][r1] = 1
    return res

# i 主元所在行，n 矩阵维，l ari/aii


def Frobinius(n, i, l: list):
    res = zeros([n, n])
    for t in range(n):
        res[t][t] = 1
    for t in range(i+1, n):
        res[t][i] = -l[t-i-1]
    return res


# U*res=c

def Hilbert(n):
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


def CholeskySolve(U, c):
    HDagger = CholeskyDecDagger(U)
    y = LSolve(HDagger, c)
    return USolve(HDagger.T(), y)
