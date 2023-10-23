import numpy as np
import pylab as pl

N = 1000000
np.random.seed(0)
x = np.random.randn(2,N)
A = np.random.randn(2,2)

y = A.dot(x)
R = y.dot(y.T)/N

print(R)
print(A.dot(A.T))
print(np.linalg.svd(R))

R2 = R * R.T
print(R2)
print(np.linalg.svd(R2))

