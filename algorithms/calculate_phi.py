import numpy as numpy

def calculate_phi(A, H,spected_result):
    phis = np.arange(A.length)
    phis[L] = spected_result - A[L]
    for i in range(0,L-1):
        phis[i] = np.matmul(H[l-1].T,phi)* a[i] * (1-a(l))
    return phis