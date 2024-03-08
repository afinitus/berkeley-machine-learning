import numpy as np
np.random.seed(1)
A = np.random.rand(5, 5)
a = np.random.rand(5)
b = np.random.rand(5)
print("Trace Difference:", np.einsum('ii->', A) - np.trace(A))
print("MatMul Diff Norm:", np.linalg.norm(np.einsum('ij,j->i', A, b) - np.matmul(A, b)))
print("OuterProduct Diff Norm:", np.linalg.norm(np.einsum('i,j->ij', a, b) - np.outer(a, b)))