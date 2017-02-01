from sympy import Matrix, Symbol, diff, cos, sin, sqrt, pprint, simplify
from scipy.integrate import  dblquad
from scipy.stats.distributions import chi2
import math
import numpy as np
from mpmath import quad
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

teta = Symbol('teta')
phi = Symbol('phi')

x = sqrt(3)*sin(teta)*cos(phi)
y = sqrt(2)*sin(teta)*sin(phi)
z = cos(teta)
J = Matrix([[diff(x, teta), diff(x, phi)], [diff(y, teta), diff(y, phi)], [diff(z, teta), diff(z, phi)]])
q = simplify(sqrt((J.T*J).det()))
p = 0.084 * q

################

q = 1
M = 0.145
N = 1000
X = lambda teta, phi: math.sqrt(3)*math.sin(teta)*math.cos(phi)
Y = lambda teta, phi: math.sqrt(2)*math.sin(teta)*math.sin(phi)
Z = lambda teta, phi: math.cos(teta)
P = lambda teta, phi: 0.084*math.sqrt(6*math.sin(teta)**2 - 4*math.sin(teta)**4 + (math.sin(phi)**2)*math.sin(teta)**4)

points = list()

#отбираем подходящие параметры
while (len(points) < N):
    _teta = np.random.sample() * math.pi/2
    _phi = np.random.sample() * 2*math.pi
    alpha = np.random.sample()
    if P(_teta, _phi) >= alpha*M*q:
        points.append((_teta,_phi))
#
# #строим график заданной поверхности
#
# TETA, PHI = np.meshgrid(np.linspace(0, math.pi/2, 100), np.linspace(0, math.pi*2, 100))
# fig = plt.figure()
# ax = fig.gca(projection = '3d')
# _X = math.sqrt(3)*np.sin(TETA)*np.cos(PHI)
# _Y = math.sqrt(2)*np.sin(TETA)*np.sin(PHI)
# _Z = np.cos(TETA)
# surf = ax.plot_surface(_X, _Y, _Z)
# #строим график из равномерных по площади точек
# _X = [X(param[0], param[1]) for param in points]
# _Y = [Y(param[0], param[1]) for param in points]
# _Z = [Z(param[0], param[1]) for param in points]
# surf = ax.scatter(_X, _Y, _Z, c = 'red')
# #равномерные по параметру
# TETA = np.random.sample(N)*math.pi/2
# PHI = np.random.sample(N)*2*math.pi
#
# _X = [X(TETA[i], PHI[i]) for i in range(N)]
# _Y = [Y(TETA[i], PHI[i]) for i in range(N)]
# _Z = [Z(TETA[i], PHI[i]) for i in range(N)]
# surf = ax.scatter(_X, _Y, _Z, c = 'green')
# plt.show()
def Pirson (param):
    K = 100
    alpha = 0.05
    N = len(param)
    s = np.linspace(0, 2*math.pi, K+1)
    T = [np.sum(s[i] < j <= s[i+1] for j in param)/N for i in range(K)] # доля точек над сектором
    Pr = []
    surface = dblquad(P, 0, np.pi / 2, lambda phi: 0, lambda phi: 2 * np.pi)
    for i in range(K):
        sector = quad(P, [0, np.pi/2], [s[i], s[i+1]])
        Pr.append(sector/surface[0])
    CHI2 = N*sum([((T[i]-Pr[i])**2)/Pr[i] for i in range(K)])
    t_alpha = chi2.ppf(1 - alpha, K-1)
    print(CHI2, t_alpha)
    return CHI2 < t_alpha
np_points = np.array(points)
print (Pirson(np_points[:,1]))

L = 100
T = []
for i in range(L):
    print (i)
    _phi = np.random.sample(N) * 2 * np.pi
    T.append(Pirson(_phi))
s = sum(T)
print (s/L)