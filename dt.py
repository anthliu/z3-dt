'''
Implements the DT->SAT algorithm from "Learning Optimal Decision Trees with SAT"
https://www.ijcai.org/proceedings/2018/0189.pdf
'''

import numpy as np
import z3
from toydata import gen_data

class SATDT(object):
    def __init__(self, N, K):
        self.N = N
        self.K = K
        self.model = None
        self.data_constraints = []
        self.constraints = []

        # initialize
        self.v = {}# leaf node
        for i in range(1, N+1):
            self.v[i] = z3.Bool(f'v{i}')
        self.l = {}# left child
        self.r = {}# right child
        self.p = {}# node parent

        # l/r child indices
        self.LR = [set(j for j in range(i+1, min(2 * i, N-1)+1) if (j % 2 == 0)) for i in range(N+1)]
        self.RR = [set(j for j in range(i+2, min(2 * i + 1, N)+1) if (j % 2 == 1)) for i in range(N+1)]
        for i in range(1, N+1):
            for j in self.LR[i]:
                self.l[i, j] = z3.Bool(f'l{i},{j}')
        for i in range(1, N+1):
            for j in self.RR[i]:
                self.r[i, j] = z3.Bool(f'r{i},{j}')
        for i in range(1, N):
            for j in range(2, N+1):
                self.p[j, i] = z3.Bool(f'p{j},{i}')

        self.a = {}# feature is assigned to node
        self.u = {}# feature is being discriminated against by node
        self.d0 = {}# feature is being discriminated for value 0 by node
        self.d1 = {}# feature is being discriminated for value 1 by node
        self.c = {}# class of leaf node
        for j in range(1, N+1):
            self.c[j] = z3.Bool(f'c{j}')
            for k in range(1, K+1):
                self.a[k, j] = z3.Bool(f'a{k},{j}')
                self.u[k, j] = z3.Bool(f'u{k},{j}')
                self.d0[k, j] = z3.Bool(f'd0,{k},{j}')
                self.d1[k, j] = z3.Bool(f'd1,{k},{j}')

        ## DT SAT constraints
        # root node is not a leaf
        self.constraints.append(z3.Not(self.v[1]))

        for i in range(1, N+1):
            for j in self.LR[i]:
                # if node is leaf, then no children
                self.constraints.append(z3.Implies(self.v[i], z3.Not(self.l[i, j])))
                # left and right child are numbered consecutively
                self.constraints.append(self.l[i, j] == self.r[i, j+1])

        for i in range(1, N+1):
            # non-leaf node must have a child
            if len(self.LR[i]) == 0:
                has_child = False
            else:
                has_child = z3.PbEq([(self.l[i, j], 1) for j in self.LR[i]], 1)
            self.constraints.append(z3.Implies(z3.Not(self.v[i]), has_child))

            # if ith node is a parent then it must have a child
            for j in self.LR[i]:
                self.constraints.append(self.p[j, i] == self.l[i, j])
            for j in self.RR[i]:
                self.constraints.append(self.p[j, i] == self.r[i, j])

        # all nodes but the first must have a parent
        for j in range(2, N+1):
            self.constraints.append(z3.PbEq([(self.p[j, i], 1) for i in range(j // 2, j)], 1))

        for k in range(1, K+1):
            # constraints to discriminate a feature for value 0
            self.constraints.append(z3.Not(self.d0[k, 1]))
            for j in range(2, N+1):
                inner = []
                for i in range(j // 2, j):
                    if (i, j) not in self.r:
                        self.r[i, j] = False
                    inner.append(z3.Or(z3.And(self.p[j, i], self.d0[k, i]), z3.And(self.a[k, i], self.r[i, j])))
                self.constraints.append(self.d0[k, j] == z3.Or(*inner))

            # constraints to discriminate a feature for value 1
            self.constraints.append(z3.Not(self.d1[k, 1]))
            for j in range(2, N+1):
                inner = []
                for i in range(j // 2, j):
                    if (i, j) not in self.l:
                        self.l[i, j] = False
                    inner.append(z3.Or(z3.And(self.p[j, i], self.d1[k, i]), z3.And(self.a[k, i], self.l[i, j])))
                self.constraints.append(self.d1[k, j] == z3.Or(*inner))

            # to use a feature at node
            for j in range(1, N+1):
                inner1 = []
                inner2 = []
                for i in range(max(j//2, 1), j):
                    inner1.append(z3.Implies(z3.And(self.u[k, i], self.p[j, i]), z3.Not(self.a[k, j])))
                    inner2.append(z3.And(self.u[k, i], self.p[j, i]))

                self.constraints.append(z3.And(*inner1))
                self.constraints.append(self.u[k, j] == z3.Or(self.a[k, j], *inner2))

        for j in range(1, N+1):
            # for a non-leaf node, exactly one feature is used
            one_feature = z3.PbEq([(self.a[k, j], 1) for k in range(1, K+1)], 1)
            self.constraints.append(z3.Implies(z3.Not(self.v[j]), one_feature))

            # for a leaf node, no feature is used
            no_feature = z3.PbEq([(self.a[k, j], 1) for k in range(1, K+1)], 0)
            self.constraints.append(z3.Implies(self.v[j], no_feature))


    def _data_constraints(self, X, y):
        assert X.shape[1] == self.K
        assert X.shape[0] == y.shape[0]
        self.data_constraints = []

        for _x, _y in zip(X, y):
            for j in range(1, self.N+1):
                inner = []
                for k in range(1, self.K+1):
                    inner.append(self.d1[k, j] if _x[k - 1] else self.d0[k, j])
                # positive examples
                if _y:
                    self.data_constraints.append(z3.Implies(z3.And(self.v[j], z3.Not(self.c[j])), z3.Or(*inner)))
                else:
                    self.data_constraints.append(z3.Implies(z3.And(self.v[j], self.c[j]), z3.Or(*inner)))

    def fit(self, X, y):
        self._data_constraints(X, y)

        s = z3.Solver()
        s.add(self.constraints + self.data_constraints)
        if s.check() == z3.sat:
            self.model = s.model()
            return True
        else:
            return False

    def parse_tree(self, node=1):
        is_leaf = self.model.evaluate(self.v[node])
        if is_leaf:
            return self.model.evaluate(self.c[node])
        for j in self.LR[node]:
            if self.model.evaluate(self.l[node, j]):
                l_child = self.parse_tree(node=j)
                break
        for j in self.RR[node]:
            if self.model.evaluate(self.r[node, j]):
                r_child = self.parse_tree(node=j)
                break
        feature = -1
        for k in range(1, self.K+1):
            if self.model.evaluate(self.a[k, node]):
                feature = k
                break

        return (feature, (l_child, r_child))

class SoftSATDT(SATDT):
    def fit(self, X, y, ws=None):
        if ws is not None:
            assert X.shape[0] == ws.shape[0]
        self._data_constraints(X, y)

        s = z3.Optimize()
        s.add(self.constraints)
        if ws is not None:
            assert len(self.data_constraints) == self.N * ws.shape[0]
            for idx in range(ws.shape[0]):
                start_idx = self.N * idx
                end_idx = self.N * idx + self.N
                s.add_soft(self.data_constraints[start_idx:end_idx], weight=int(ws[idx]), id=f'err{idx}')
        else:
            s.add_soft(self.data_constraints, weight=1, id=f'err')
        if s.check() == z3.sat:
            self.model = s.model()
            self.error = sum(self.model.evaluate(obj).as_long() for obj in s.objectives())
            return True
        else:
            return False

def test():
    X, y = gen_data(200, 8, lambda x: (x[:,0] & x[:,1]) | (x[:,2] & x[:,3] & x[:,4]))
    #X, y = gen_data(42, 4, lambda x: (x[:,0] & x[:,1]) | x[:, 2])

    for nodes in range(10, 20):
        dt = SoftSATDT(nodes, X.shape[1])
        if dt.fit(X, y):
            #print(dt.model)
            print(dt.parse_tree())
            print(f'Nodes {nodes} error: {dt.error}')
        else:
            print(f'failed to solve for nodes = {nodes}')

if __name__ == '__main__':
    test()
