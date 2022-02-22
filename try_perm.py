import numpy as np

def revert_permutation(perm: np.ndarray) -> np.ndarray:
    perm_rev = np.arange(perm.shape[0])
    perm_rev[perm] = np.copy(perm_rev)
    return perm_rev

N = 10
M = 3
np.random.seed(0)

mat = np.zeros((N, M))

for i in range(0,N):
  for j in range (0,M):
    mat[i, j] = i*M+j

perm = np.arange(10)
np.random.shuffle(perm)

perm_mat = np.take(mat, perm, axis=0)
manual_perm_mat = np.zeros((N,M))
for i in range(0,N):
  manual_perm_mat[i,:] = mat[perm[i],:]

assert np.all(manual_perm_mat == perm_mat)  

rev_perm = revert_permutation(perm)
print(perm)
print(rev_perm)