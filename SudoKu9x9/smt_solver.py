from nn_utils import *
import numpy as np
from z3 import *
import time
from joblib import Parallel, delayed


def sample_search(probability, threshold=0.5):
    max_iter = len(probability)
    def search_(idx):
        latent = probability[idx,:]
        prob = latent.clone().cpu().detach()
        max_prob, pred = torch.max(prob, dim=-1)
        ind = torch.where(max_prob < threshold)
        tmp = pred.clone()
        tmp[ind] = 0
        sat, sol = check(tmp)
        # print(tmp, sat)
        if sat == False:
            return
        if sat == True:
            pred = torch.Tensor(sol).long()
            return idx, pred.unsqueeze(dim=0)
    pseudos = Parallel(n_jobs=10)(delayed(search_)(idx) for idx in range(max_iter))
    train_X = []
    train_Y = []
    for p in pseudos:
        if p is not None:
            idx = p[0]
            train_X.append(probability[idx].unsqueeze(dim=0))
            train_Y.append(p[1])

    if len(train_X) > 0 and len(train_Y) > 0:
        train_X = torch.cat(train_X, dim=0).cuda()
        train_Y = torch.cat(train_Y, dim=0).long().cuda()
        return train_X, train_Y
    else:
        return None, None

def check(sol):
    filename = "./data/z3solve.smt2"
    t1 = time.time()
    X = [[Int("X_%s_%s" % (i+1, j+1)) for j in range(size)] for i in range(size) ]
    # check
    s = SolverFor("QF_FD")
    s.reset()

    constraints = parse_smt2_file(filename, sorts={}, decls={})
    s.add(constraints)

    # now we put the assumptions of the given puzzle into the solver:
    index = np.where(sol >= 1)
    for i, j in zip(index[0], index[1]):
        s.add(X[i][j] == int(sol[i,j]))

    if s.check() == sat:
        return True, [[s.model()[X[i][j]].as_long() for j in range(size)] for i in range(size)]
    else:
        return False, None


def init_check(sol):
    filename = "./data/z3solve.smt2"
    # check
    s = SolverFor("QF_FD")
    # set_option("parallel.enable", True)
    X = [[Int("X_%s_%s" % (i+1, j+1)) for j in range(size)] for i in range(size) ]


    for i in range(size):
        for j in range(size):
            s.add(Or([X[i][j] == k for k in range(1, size+1)]))

    # #### assure that every row covers every value:
    for c in range(size):
        s.add(Distinct([X[c][i] for i in range(size)]))
        s.add(Distinct([X[i][c] for i in range(size)]))

    # #### assure that every block covers every value:
    for i in range(3):
        for j in range(3):
            s.add(Distinct([X[m + i * 3][n + j * 3] for m in range(3) for n in range(3)]))

    # save state
    with open(filename, mode='w') as f:
        f.write(s.to_smt2())

    # now we put the assumptions of the given puzzle into the solver:
    index = np.where(sol >= 1)
    for i, j in zip(index[0], index[1]):
        s.add(X[i][j] == int(sol[i,j]))

    # t1 = time.time()

    if s.check() == sat:
        # t2 = time.time()
        # print(t2-t1)
        return True, [[s.model()[X[i][j]].as_long() for j in range(size)] for i in range(size)]
    else:
        return False, None












