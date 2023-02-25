from nn_utils import *
import numpy as np
from z3 import *
from graphs import Graph
import copy
import networkx as nx
import time

def sample_search(preds, batch_idx):
    train_X = []
    train_Y = []
    for latent, idx in zip(preds, batch_idx):
        pred = latent.clone().cpu().detach().round().long()
        preds = pred.data.tolist()

        sat, _ = r_check(preds, idx)
        if sat == True:
            train_X.append(latent.unsqueeze(dim=0))
            train_Y.append(pred.unsqueeze(dim=0))

    if len(train_X) > 0 and len(train_Y) > 0:
        train_X = torch.cat(train_X, dim=0).cuda()
        train_Y = torch.cat(train_Y, dim=0).float().cuda()
        return train_X, train_Y
    else:
        return None, None

def r_check(preds, idx):
    # t2 = time.time()
    filename = "./data/smt/z3solve_{}.smt2".format(idx)
    s = SolverFor("QF_LIA")
    s.reset()
    Z = IntVector('z', n)
    constraints = parse_smt2_file(filename, sorts={}, decls={})
    s.add(constraints)

    for i in range(n):
        s.add(preds[i] == Z[i])
    
    if s.check() == sat:
        # t3 = time.time()
        # print(t3-t2)
        return True, [float(s.model()[z].as_long()) for z in Z]
    else:
        return False, None
    

def check(preds, idx):
    # t2 = time.time()
    filename = "./data/smt/z3solve_{}.smt2".format(idx)
    s = SolverFor("QF_LIA")
    s.reset()
    Z = IntVector('z', n)
    constraints = parse_smt2_file(filename, sorts={}, decls={})
    s.add(constraints)

    for i in range(n):
        if i % 6 != 0:
            ind = ((i-1)//6)*5 + (i-1)%6
            s.add(preds[ind] == Z[i])
    
    if s.check() == sat:
        # t3 = time.time()
        # print(t3-t2)
        return True, [float(s.model()[z].as_long()) for z in Z]
    else:
        return False, None
    
    
def init_check(graph, idx, radius):
    g = graph.g
    inp = graph.input
    filename = "./data/smt/z3solve_{}.smt2".format(idx)
    s = SolverFor("QF_LIA")
    Z = IntVector('z', graph.n)
    for i in range(graph.n):
        s.add(Z[i]>=0)
    appro_d = []
    for i in range(graph.n):
        appro_d.append(Z[i])
    for i in range(graph.n):
        paths = copy.deepcopy(graph.paths[str(i)])
        cons = []
        for path in paths:
            j = path.pop(0)
            c = []
            while j != 0:
                k = path.pop(0)
                index = np.where(inp[j,:] != 0)[0]
                for t in index:
                    flag = False
                    for tmp in graph.paths[str(i)]:
                        if j in tmp:
                            ind = tmp.index(j)
                            if ind < len(tmp)-1 and tmp[ind + 1] == t:
                                flag = True
                                break
                        else:
                            break
                    if flag:
                        continue
                    else:
                        c.append(appro_d[k] - appro_d[t] < - inp[k,j] + inp[t,j] - radius) 
                j = k
            cons.append(And(c))
        s.add(Or(cons))

    # save state
    with open(filename, mode='w') as f:
        f.write(s.to_smt2())
        
    if s.check() == sat:
        return True, [float(s.model()[z].as_long()) for z in Z]
    else:
        return False, None
    


if __name__ == "__main__":
    n = 30; m = 100
    graph = Graph.gen_random_graph(n, m)
    # print(graph.g.edges)
    sat, preds = init_check(graph)
    print(sat, preds)
    paths, dists = eval_pred(graph.input, np.array(preds)+5)
    for i in range(graph.n):
        p_hat = paths[str(i)]
        p = graph.paths[str(i)]
        print(p_hat in p, dists[i] == graph.d[i,0])
    print(preds, graph.d[0,:])











