import numpy as np
import scipy


# Figure 3.1, AN ARNOLDI-TYPE ALGORITHM FOR COMPUTING PAGE RANK, G. H. GOLUB1, and C. GREIF, DOI: 10.1007/s10543-006-0091-y
def arnoldi(A, n):
    b = np.ones(A.shape[0])
    for i in range(99999):
        #print("arnnnn", i)
        h, Q = arnoldi_n(A, b, n)
        vals, hvecs = eig(h)
        vecs = Q.dot(hvecs)
        b = vecs[:, 0]
        diff = np.linalg.norm(A.dot(b) - vals[0]*b, 2)
        if diff < 1e-6:
            break

    return vals, vecs


def arnoldi_n(A, b, n):
    eps = 1e-12
    h = np.zeros((n + 1, n), dtype=A.dtype)
    Q = np.zeros((A.shape[0], n + 1), dtype=A.dtype)
    # Normalize the input vector
    Q[:, 0] = b / np.linalg.norm(b, 2)  # Use it as the first Krylov vector
    nK = -1
    for k in range(1, n + 1):
        nK = k
        v = np.dot(A, Q[:, k - 1])  # Generate a new candidate vector
        for j in range(k):  # Subtract the projections on previous vectors
            h[j, k - 1] = np.dot(Q[:, j].conj(), v)
            v = v - h[j, k - 1] * Q[:, j]
        h[k, k - 1] = np.linalg.norm(v, 2)
        if h[k, k - 1] > eps:  # Add the produced vector to the list, unless
            Q[:, k] = v / h[k, k - 1]
        else:  # If that happens, stop iterating.
            break

    h = h[:nK, :nK]
    Q = Q[:,:nK]
    return h, Q


def lanczos(A, m):
    n = A.shape[0]
    v = np.random.random(n).astype(np.complex64)
    v = v / np.linalg.norm(v)
    v_old = np.zeros(n, dtype=np.complex64)
    beta = np.zeros(m, dtype=np.complex64)
    alpha = np.zeros(m, dtype=np.complex64)
    for j in range(m-1):
        w = A.dot(v)
        alpha[j] = np.conj(w).T.dot(v)
        w = w - alpha[j] * v - beta[j] * v_old
        beta[j+1] = np.linalg.norm(w)
        v_old = v.copy()
        v = w / beta[j+1]
    w = A.dot(v)
    alpha[m-1] = np.conj(w).T.dot(v)
    A = np.diag(beta[1:], k=-1) + np.diag(beta[1:], k=1) + np.diag(alpha[:], k=0)
    l, _ = np.linalg.eigh(A)
    return l


def eig(A):
    vals, vecs = scipy.linalg.eig(A)
    eig_idx = vals.argsort()
    vals = vals[eig_idx]
    vecs = vecs[:,eig_idx]
    return vals, vecs


A = np.array([
    [-3j, -1, -1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
	[-1, -1, 0, -1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
	[-1, 0, 1, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
	[0, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0],
	[-1, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0],
	[0, -1, 0, 0, -1, 3, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0],
	[0, 0, -1, 0, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, -1, 0],
	[0, 0, 0, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1],
	[-1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 0, 0, 0],
	[0, -1, 0, 0, 0, 0, 0, 0, -1, 1, 0, -1, 0, -1, 0, 0],
	[0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 3, -1, 0, 0, -1, 0],
	[0, 0, 0, -1, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, -1],
	[0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, 0],
	[0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 1, 0, -1],
	[0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, -1, 0, -1, -1],
	[0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, -1, -1, -3],
    ], dtype=np.complex64)


print("largest vec", np.linalg.norm(eig(A)[1][:,0], 2), eig(A)[1][:,0])
print("exact", eig(A)[0])
l_large_exact = eig(A)[0][0]

avals, avecs = arnoldi(A, 2)
avecs *= eig(A)[1][:,0][0]/avecs[:,0][0]
print("arnoldi vals", avals)
print("arnoldi vec", avecs[:,0])

scipyVal, scipyVec = scipy.sparse.linalg.eigs(A)
scipyVec *= eig(A)[1][:,0][0]/scipyVec[:,0][0]
print("scipyVal", scipyVal[0])
print("scipyVec", np.linalg.norm(scipyVec[:, 0], 2), scipyVec[:, 0])

#l_large_exact = scipy.linalg.eigh(A)[0][0]
print("largegggg", l_large_exact)
print('k=10, err = {}'.format(np.abs(l_large_exact - arnoldi(A, 10)[0][0])))
