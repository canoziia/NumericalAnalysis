from typing import *
import time


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


def sameType(x):
    if hasattr(x, "sameType"):
        return x.sameType()
    if isNum(x):
        return 0
    else:
        raise UnknownType


class vector(list):
    def __init__(self, arr):
        list.__init__([])
        self.extend(arr)

    def sameType(self):
        # res=self.__class__([])
        # for i in self:
        #     res.append(i.sameType())
        return vector([0]*len(self))

    def conjugate(self):
        res = self.__class__([])
        for i in range(len(self)):
            res.append(conjugate(self[i]))
        return res

    def Conjugate(self) -> None:
        for i in range(len(self)):
            self[i] = conjugate(self[i])

    def chklen(self, x):
        if len(self) != len(x):
            raise LengthMisMatch

    def __add__(self, x):
        if isNum(x):
            return self.__class__([(self[i]+x) for i in range(len(self))])
        self.chklen(x)
        return self.__class__([(self[i]+x[i]) for i in range(len(self))])

    def __radd__(self, x):
        return self.__add__(x)

    def __iadd__(self, x):
        return self.__add__(x)

    def __sub__(self, x):
        if isNum(x):
            return self.__class__([(self[i]-x) for i in range(len(self))])
        self.chklen(x)
        return self.__class__([(self[i]-x[i]) for i in range(len(self))])

    def __neg__(self):
        return self.__class__([(-self[i]) for i in range(len(self))])

    def __pos__(self):
        return self.__class__([(+self[i]) for i in range(len(self))])

    def __rsub__(self, x):
        if isNum(x):
            return self.__class__([(x-self[i]) for i in range(len(self))])
        self.chklen(x)
        return self.__class__([(x[i]-self[i]) for i in range(len(self))])
    # 标积

    def __mul__(self, x):
        if isNum(x):
            return vector([(x*i) for i in self])
        self.chklen(x)
        res = sameType(x[0])
        for i in range(len(self)):
            res += self[i]*x[i]
        return res

    def __rmul__(self, x):
        return self.__mul__(x)


class mat(vector):
    def __init__(self, arr):
        list.__init__([])
        self.extend(arr)

    def row(self):
        return len(self)

    def col(self):
        return len(self[0])

    def createRes(self, x):
        if isNum(x) or type(x) == mat:
            return mat([])
        elif type(x) == vector:
            return vector([])

    def extend(self, arr) -> None:
        for x in arr:
            self.append(self.__class__.__base__(x))

    def chklen(self, x):
        if self.col() != len(x):
            raise LengthMisMatch

    def appendRow(self, arr: list):
        if self.row() != len(arr):
            raise LengthMisMatch
        for i in range(self.row()):
            self[i].append(arr[i])

    def changeRow(self, r1, r2):
        # t = self[r1]
        # self[r1] = self[r2]
        # self[r2] = t
        if r1 == r2:
            return
        self[r1], self[r2] = self[r2], self[r1]

    def changeCol(self, c1, c2):
        # t=0
        if c1 == c2:
            return
        for i in range(self.col()):
            # t=self[i][c1]
            self[i][c1], self[i][c2] = self[i][c2], self[i][c1]

    def T(self):
        res = self.__class__([])
        # oldrow
        orow = self.row()
        if orow == 0:
            return res
        for nrow in range(self.col()):
            res.append(self.__class__.__base__([]))
            for ncolomn in range(orow):
                res[nrow].append(self[ncolomn][nrow])
        return res

    def __mul__(self, x):
        r1 = self.row()
        if r1 == 0:
            raise LengthMisMatch
        res = self.createRes(x)
        for i in range(r1):
            res.append(self[i]*x)
        return res


def zeros(row, colomn):
    return mat([[0]*colomn for _ in range(row)])


def PmtMat(n, r1, r2):
    res = zeros(n, n)
    for i in range(n):
        res[i][i] = 1
    res[r1][r1] = 0
    res[r2][r2] = 0
    res[r1][r2] = 1
    res[r2][r1] = 1
    return res

# i 主元所在行，n 矩阵维，l ari/aii


def Frobinius(n, i, l: list):
    res = zeros(n, n)
    for t in range(n):
        res[t][t] = 1
    for t in range(i+1, n):
        res[t][i] = -l[t-i-1]
    return res


# U*res=c

def Hilbert(n):
    res = zeros(n, n)
    for i in range(n):
        for j in range(n):
            res[i][j] = 1/(i+j+1)
    return res


def USolve(U: mat, c: vector) -> vector:
    n = len(U)
    res = vector([0] * n)
    for i in range(n-1, -1, -1):
        t = 0
        for k in range(i+1, n):
            t += U[i][k]*res[k]
        res[i] = (c[i]-t)/U[i][i]
    return res


def LSolve(L: mat, c: vector) -> vector:
    n = len(L)
    res = vector([0]*n)
    for i in range(n):
        t = 0
        for k in range(i):
            t += L[i][k]*res[k]
        res[i] = (c[i]-t)/L[i][i]
    return res


def GEM(U: mat, c: vector) -> vector:
    U.appendRow(c)
    row = len(U)
    # col=row+1
    col = len(U[0])
    # i主元所在行
    i = 0
    for i in range(row):
        for it in range(i, row):
            if U[it][i] != 0:
                U.changeRow(it, i)
        l = [U[t][i]/U[i][i] for t in range(i+1, row)]
        U = Frobinius(row, i, l)*U
    # res = vector([0] * row)
    # for i in range(row-1, -1, -1):
    #     t = 0
    #     for k in range(i+1, row):
    #         t += U[i][k]*res[k]
    #     res[i] = (U[i][col-1]-t)/U[i][i]
    ctmp = vector([])
    for i in range(row):
        ctmp.append(U[i][col-1])
        U[i].pop()
    return USolve(U, ctmp)


def CholeskyDecDagger(U: mat) -> mat:
    if len(U) == 0 or len(U) != len(U[0]):
        raise LengthMisMatch
    n = len(U)
    H = zeros(n, n)
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


def CholeskyDec(U: mat) -> mat:
    return CholeskyDecDagger(U).T().conjugate()


def CholeskySolve(U, c):
    HDagger = CholeskyDecDagger(U)
    y = LSolve(HDagger, c)
    return USolve(HDagger.T(), y)
