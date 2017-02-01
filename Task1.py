import numpy as np
def stat(N, M, x, y):
    p_est = np.array([i/N for i in x])
    q_est = np.array([i/M for i in y])
    r_est = (x + y)/(N + M)
    return np.sum(x*np.log(p_est)) + np.sum(y*np.log(q_est)) - np.sum((x + y)*np.log(r_est))
x = np.array([4782869, 1869615, 907564, 93048, 56154])
y = np.array([5188154, 1034165, 210927, 131377, 90280])
N = np.sum(x)
M = np.sum(y)
L = stat(N, M, x, y)
alpha = 0.01
K = 1000
P = (x + y)/(N + M)
T = []
for i in range(K):
    x_sample = np.random.multinomial(N, P)
    y_sample = np.random.multinomial(M, P)
    T.append(stat(N, M, x_sample, y_sample))
T.sort()
t_alpha = T[round((1-alpha)*K)]
print (t_alpha, L)