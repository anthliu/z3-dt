'''
Implements the DT->SAT algorithm from "Learning Optimal Decision Trees with SAT"
https://www.ijcai.org/proceedings/2018/0189.pdf
'''

import z3
from toydata import gen_data

def dt_sat_constraints(N, K, X, y):
    assert X.shape[1] == K
    assert X.shape[0] == y.shape[0]
    # initialize
    v = {}# leaf node
    for i in range(1, N+1):
        v[i] = z3.Bool(f'v{i}')
    l = {}# left child
    r = {}# right child
    p = {}# node parent

    # l/r child indices
    LR = [set(j for j in range(i+1, min(2 * i, N-1)+1) if (j % 2 == 0)) for i in range(N+1)]
    RR = [set(j for j in range(i+2, min(2 * i + 1, N)+1) if (j % 2 == 1)) for i in range(N+1)]
    for i in range(1, N+1):
        for j in LR[i]:
            l[i, j] = z3.Bool(f'l{i},{j}')
    for i in range(1, N+1):
        for j in RR[i]:
            r[i, j] = z3.Bool(f'r{i},{j}')
    for i in range(1, N):
        for j in range(2, N+1):
            p[j, i] = z3.Bool(f'p{j},{i}')
    
    constraints = []
    # root node is not a leaf
    constraints.append(z3.Not(v[1]))

    for i in range(1, N+1):
        for j in LR[i]:
            # if node is leaf, then no children
            constraints.append(z3.Implies(v[i], z3.Not(l[i, j])))
            # left and right child are numbered consecutively
            constraints.append(l[i, j] == r[i, j+1])

    for i in range(1, N+1):
        # non-leaf node must have a child
        if len(LR[i]) == 0:
            has_child = False
        else:
            has_child = z3.PbEq([(l[i, j], 1) for j in LR[i]], 1)
        constraints.append(z3.Implies(z3.Not(v[i]), has_child))

        # if ith node is a parent then it must have a child
        for j in LR[i]:
            constraints.append(p[j, i] == l[i, j])
        for j in RR[i]:
            constraints.append(p[j, i] == r[i, j])

    # all nodes but the first must have a parent
    for j in range(2, N+1):
        constraints.append(z3.PbEq([(p[j, i], 1) for i in range(j // 2, j)], 1))

    a = {}# feature is assigned to node
    u = {}# feature is being discriminated against by node
    d0 = {}# feature is being discriminated for value 0 by node
    d1 = {}# feature is being discriminated for value 1 by node
    c = {}# class of leaf node
    for j in range(1, N+1):
        c[j] = z3.Bool(f'c{j}')
        for k in range(1, K+1):
            a[k, j] = z3.Bool(f'a{k},{j}')
            u[k, j] = z3.Bool(f'u{k},{j}')
            d0[k, j] = z3.Bool(f'd0,{k},{j}')
            d1[k, j] = z3.Bool(f'd1,{k},{j}')

    for k in range(1, K+1):
        # constraints to discriminate a feature for value 0
        constraints.append(z3.Not(d0[k, 1]))
        for j in range(2, N+1):
            inner = []
            for i in range(j // 2, j):
                if (i, j) not in r:
                    r[i, j] = False
                inner.append(z3.Or(z3.And(p[j, i], d0[k, i]), z3.And(a[k, i], r[i, j])))
            constraints.append(d0[k, j] == z3.Or(*inner))

        # constraints to discriminate a feature for value 1
        constraints.append(z3.Not(d1[k, 1]))
        for j in range(2, N+1):
            inner = []
            for i in range(j // 2, j):
                if (i, j) not in l:
                    l[i, j] = False
                inner.append(z3.Or(z3.And(p[j, i], d1[k, i]), z3.And(a[k, i], l[i, j])))
            constraints.append(d1[k, j] == z3.Or(*inner))

        # to use a feature at node
        for j in range(1, N+1):
            inner1 = []
            inner2 = []
            for i in range(max(j//2, 1), j):
                inner1.append(z3.Implies(z3.And(u[k, i], p[j, i]), z3.Not(a[k, j])))
                inner2.append(z3.And(u[k, i], p[j, i]))

            constraints.append(z3.And(*inner1))
            constraints.append(u[k, j] == z3.Or(a[k, j], *inner2))
            
    for j in range(1, N+1):
        # for a non-leaf node, exactly one feature is used
        one_feature = z3.PbEq([(a[k, j], 1) for k in range(1, K+1)], 1)
        constraints.append(z3.Implies(z3.Not(v[j]), one_feature))

        # for a leaf node, no feature is used
        no_feature = z3.PbEq([(a[k, j], 1) for k in range(1, K+1)], 0)
        constraints.append(z3.Implies(v[j], no_feature))

    data_cons = []
    # positive examples
    for x in X[y]:
        for j in range(1, N+1):
            inner = []
            for k in range(1, K+1):
                inner.append(d1[k, j] if x[k - 1] else d0[k, j])
            data_cons.append(z3.Implies(z3.And(v[j], z3.Not(c[j])), z3.Or(*inner)))

    # negative examples
    for x in X[~y]:
        for j in range(1, N+1):
            inner = []
            for k in range(1, K+1):
                inner.append(d1[k, j] if x[k - 1] else d0[k, j])
            data_cons.append(z3.Implies(z3.And(v[j], c[j]), z3.Or(*inner)))

    return constraints + data_cons

def test():
    X, y = gen_data(100, 8, lambda x: (x[:,0] & x[:,1]) | (x[:,2] & x[:,3] & x[:,4]))

    for nodes in range(10, 20):
        constraints = dt_sat_constraints(nodes, 8, X.copy(), y.copy())
        s = z3.Solver()
        s.add(constraints)
        if s.check() == z3.sat:
            m = s.model()
            print(m)
        else:
            print(f'failed to solve for nodes = {nodes}')

if __name__ == '__main__':
    test()
