cimport numpy as cnp
cimport cython
import numpy as np

cdef _ceval_path(cnp.ndarray[cnp.int32_t, ndim=3] adjs, cnp.ndarray[cnp.float32_t, ndim=2] preds, int n):
    cdef list paths = []
    cdef list dists = []
    for idx, (adj, pred) in enumerate(zip(adjs, preds)):
        dist = []
        path = {str(i): None for i in range(n)}
        for i in range(n):
            j = i
            d = 0
            p = [i] # path
            v = [i] # visited node
            while j != 0:
                index = np.where(adj[j,:] != 0)[0]
                index = list(set(index) - set(v))
                if len(index) > 1:
                    k = np.argmin(adj[j, index]+pred[index])
                elif len(index) == 1:
                    k = 0
                else:
                    break
                d += adj[j, index[k]]
                j = index[k]
                p.append(j)
                v.append(j)
            dist.append(d)
            path[str(i)] = p
        paths.append(path)
        dists.append(dist)
    return paths, dists

def ceval_path(adjs, preds, n):
    return _ceval_path(adjs, preds, n)