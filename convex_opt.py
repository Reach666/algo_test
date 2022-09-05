import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt

# Problem data.
N = 64
M = 8
L = 50
np.random.seed(1)
# A = np.random.randn(M, N)
# b = np.random.randn(M)
m = np.arange(M).reshape(-1,1)
w = (2 * np.pi * np.arange(N) / N)
A = np.exp(1j * w * m)
x_true = np.zeros([N,L])*1j
x_true[0*8+0] = 1 #np.random.randn(L)*1
x_true[1*8-2] = 0.5 #np.random.randn(L)*0.6+0.0j
noise = (np.random.randn(M,L)+1j*np.random.randn(M,L))/100
noise = 0
b = A @ x_true + noise

# N_true = N*4
# w_true = (2 * np.pi * np.arange(N_true) / N_true)
# A_true = np.exp(1j * w_true * m)
# x_true = np.zeros([N_true,L])*1j
# x_true[0*8*4+1] = 1
# x_true[1*8*4-1] = 0.9+0.0j
# noise = (np.random.randn(M,L)+1j*np.random.randn(M,L))/50
# b = A_true @ x_true + noise


# gamma must be nonnegative due to DCP rules.
gamma = cvx.Parameter(nonneg=True)
# Construct the problem.
x = cvx.Variable((N,L))
error = cvx.sum_squares(A @ x - b)
obj = cvx.Minimize(error + gamma * cvx.norm(cvx.norm(x,2,axis=1), 1))
# obj = cvx.Minimize(error + gamma * cvx.norm(cvx.norm(x,1,axis=0), 2))
prob = cvx.Problem(obj)

# Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
sq_penalty = []
l1_penalty = []
x_values = []
gamma_vals = np.logspace(-4, 6) #-4,6
for val in gamma_vals:
    gamma.value = val
    prob.solve(verbose=False)
    # Use expr.value to get the numerical value of
    # an expression in the problem.
    sq_penalty.append(error.value)
    l1_penalty.append(cvx.norm(x, 1).value)
    x_values.append(x.value)


# Plot trade-off curve.
plt.subplot(121)
plt.plot(l1_penalty, sq_penalty)
plt.xlabel(r'|x|_1', fontsize=16)
plt.ylabel(r'|Ax-b|^2', fontsize=16)
plt.title('Trade-Off Curve for LASSO', fontsize=16)

# Plot entries of x vs. gamma.
plt.subplot(122)
for i in range(N):
    plt.plot(gamma_vals, [xi[i] for xi in x_values])
plt.xlabel(r'γ', fontsize=16)
plt.ylabel(r'x_{i}', fontsize=16)
plt.xscale('log')
plt.title(r'Entries of x vs. γ', fontsize=16)

# plt.tight_layout()
plt.show()



# import cvxpy as cvx
# #定义优化变量
# x = cvx.Variable()
# y = cvx.Variable()
# # 定义约束条件
# constraints = [x + y == 1,
#                x - y >= 1]
# # 定义优化问题
# obj = cvx.Minimize((x - y)**2)
# # 定义优化问题
# prob = cvx.Problem(obj, constraints)
# #求解问题
# prob.solve()                      #返回最优值
# print("status:", prob.status)     #求解状态
# print("optimal value", prob.value) #目标函数优化值
# print("optimal var", x.value, y.value) #优化变量的值，相应变量加.value


