import na
import math
import matplotlib.pyplot as plt


def CholeskySolve(U, c):
    HDagger = na.CholeskyDecDagger(U)
    y = na.LSolve(HDagger, c)
    # print(HDagger.T().conjugate()-HDagger.T())
    return na.USolve(HDagger.T().conjugate(), y)


# n = 10
# print(na.GEM(na.Hilbert(n), na.tensor([1 for _ in range(n)])))
# print(CholeskySolve(na.Hilbert(n), na.tensor([1 for _ in range(n)])))

def Res(n):
    reslist = []
    reslist.append([""])
    reslist.append(na.tensor([1]))
    reslist.append(na.tensor([-2, 6]))
    reslist.append(na.tensor([3, -24, 30]))
    reslist.append(na.tensor([-4, 60, -180, 140]))
    reslist.append(na.tensor([5, -120, 630, -1120, 630]))
    reslist.append(na.tensor([-6, 210, -1680, 5040, -6300, 2772]))
    reslist.append(na.tensor([7, -336, 3780, -16800, 34650, -33264, 12012]))
    reslist.append(
        na.tensor([-8, 504, -7560, 46200, -138600, 216216, -168168, 51480]))
    reslist.append(na.tensor(
        [9, -720, 13860, -110880, 450450, -1009008, 1261260, -823680, 218790]))
    reslist.append(na.tensor(
        [-10, 990, -23760, 240240, -1261260, 3783780, -6726720, 7001280, -3938220, 923780]))
    return reslist[n]


gemError = []
choError = []

for n in range(1, 11):
    tg = (Res(n)-na.GEM(na.Hilbert(n),
          na.tensor([1 for _ in range(n)])))/Res(n)
    gemError.append(tg*tg)
    tc = (Res(n)-CholeskySolve(na.Hilbert(n),
          na.tensor([1 for _ in range(n)])))/Res(n)
    choError.append(tc*tc)

print(gemError)
print(choError)
plt.plot(range(1, 11), gemError)
plt.plot(range(1, 11), choError)
plt.savefig("res.jpg")
