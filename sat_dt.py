'''
Implements the DT->SAT algorithm from "Learning Optimal Decision Trees with SAT"
https://www.ijcai.org/proceedings/2018/0189.pdf
'''

import numpy as np
import z3
from toydata import gen_data

class SATDT(object):
    def __init__(self, N, K, additional_constraints=True):
        self.N = N
        self.K = K
        self.model = None
        self.data_constraints = []
        self.constraints = []
        self.additional_constraints = additional_constraints

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

        ## DT SAT additional constraints
        if not additional_constraints:
            return

        # additional constraints
        self.lbda = {}# upper bound on descendents
        self.gamma = {}# lower bound on descendents
        self.lbda[0, 0] = False
        self.gamma[0, 0] = False
        for i in range(1, N+1):
            self.lbda[0, i] = True
            self.gamma[0, i] = True
            for j in range(1, (N//2)+1):
                self.lbda[j, i] = z3.Bool(f'lbda,{j},{i}')
                self.lbda[j, 0] = False
            for t in range(1, N+1):
                self.gamma[t, i] = z3.Bool(f'gamma,{t},{i}')
                self.gamma[t, 0] = False

        # upper bound on descendents
        for i in range(1, N+1):
            for t in range(1, (N//2)+1):
                #self.constraints.append(self.lbda[t, i] == z3.And(z3.Or(self.lbda[t, i - 1], self.lbda[t - 1, i - 1]), self.v[i]))
                self.constraints.append(self.lbda[t, i] == z3.Or(self.lbda[t, i - 1], z3.And(self.lbda[t - 1, i - 1], self.v[i])))
                l_descendent = self.l.get((i, 2 * (i - t + 1)), False)
                r_descendent = self.r.get((i, 2 * (i - t + 1) + 1), False)
                self.constraints.append(z3.Implies(self.lbda[t, i], z3.And(z3.Not(l_descendent), z3.Not(r_descendent))))
        # lower bound on descendents
        for i in range(1, N+1):
            for t in range(1, i+1):
                #self.constraints.append(self.gamma[t, i] == z3.And(z3.Or(self.gamma[t, i - 1], self.gamma[t - 1, i - 1]), z3.Not(self.v[i])))
                self.constraints.append(self.gamma[t, i] == z3.Or(self.gamma[t, i - 1], z3.And(self.gamma[t - 1, i - 1], z3.Not(self.v[i]))))

            for t in range(np.int_(np.ceil(i / 2)), i + 1):
                l_descendent = self.l.get((i, 2 * (t - 1)), False)
                r_descendent = self.r.get((i, 2 * t - 1), False)
                self.constraints.append(z3.Implies(self.gamma[t, i], z3.And(z3.Not(l_descendent), z3.Not(r_descendent))))

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
            self.error = 0.0
            self.tree = self._parse_tree()
            self.sop = self._to_sop(self.tree, self.K)
            return True
        else:
            return False

    def _parse_tree(self, node=1):
        is_leaf = self.model.evaluate(self.v[node])
        if is_leaf:
            return self.model.evaluate(self.c[node])
        for j in self.LR[node]:
            if self.model.evaluate(self.l[node, j]):
                l_child = self._parse_tree(node=j)
                break
        for j in self.RR[node]:
            if self.model.evaluate(self.r[node, j]):
                r_child = self._parse_tree(node=j)
                break
        feature = -1
        for k in range(1, self.K+1):
            if self.model.evaluate(self.a[k, node]):
                feature = k
                break

        return (feature, (l_child, r_child))

    def _to_sop(self, root, num_feats):
        if not isinstance(root, tuple):
            return None# is leaf node
        all_sop = []
        empty_sop = np.zeros(num_feats, dtype=np.int_)
        nodes = [(empty_sop, root)]
        while len(nodes) > 0:
            sop, node = nodes.pop()
            if not isinstance(node, tuple):
                if node:
                    all_sop.append(sop)
                continue
            left_sop = sop.copy()
            right_sop = sop.copy()

            feature, (left, right) = node
            left_sop[feature - 1] = -1
            right_sop[feature - 1] = 1

            nodes.append((left_sop, left))
            nodes.append((right_sop, right))
        return np.stack(all_sop)

    def predict(self, X):
        y = []
        for x in X:
            pointer = tuple(self.tree)
            while isinstance(pointer, tuple):
                pointer = pointer[1][int(x[pointer[0] - 1])]
            y.append(pointer)
        return np.array(y)

class SoftSATDT(SATDT):
    def _int_approx(self, ws, err=1e-2):
        norm_ws = ws / ws.sum()
        coef = 1 / ws.min()
        for i in range(30):
            approx = np.rint(ws * coef).astype(np.int_)
            if np.max(np.abs((approx / approx.sum()) - norm_ws)) < err:
                break
            coef *= 2
        return approx
    def _get_weight_groups(self, ws, eps=1e-3):
        centers = np.zeros(0)
        members = []
        for i, w in enumerate(ws):
            scores = np.abs(centers - w)
            if np.any(scores < eps):
                group = scores.argmin()
                members[group].append(i)
            else:
                centers = np.append(centers, w)
                members.append([i])
        approx_centers = self._int_approx(centers)
        return list(zip(members, approx_centers))

    def fit(self, X, y, ws=None):
        if ws is not None:
            assert ws.min() > 0
            gws = self._get_weight_groups(ws)
        else:
            gws = [(list(range(X.shape[0])), 1)]

        assert X.shape[0] == sum(len(gw[0]) for gw in gws)
        assert set(range(X.shape[0])) == set().union(*(gw[0] for gw in gws))
        self._data_constraints(X, y)

        s = z3.Optimize()
        s.add(self.constraints)

        assert len(self.data_constraints) == self.N * sum(len(gw[0]) for gw in gws)
        for idx, (group, w) in enumerate(gws):
            sub_constraints = []
            for i in group:
                lower = i * self.N
                upper = (i + 1) * self.N
                sub_constraints.extend(self.data_constraints[lower:upper])
            s.add_soft(sub_constraints, weight=int(w), id=f'err{idx}')
        #s.add_soft(self.data_constraints, weight=1, id='err')

        total_gws = sum(len(gw[0]) * gw[1] for gw in gws)
        if s.check() == z3.sat:
            self.model = s.model()
            self.error = sum(self.model.evaluate(obj).as_long() / total_gws for obj in s.objectives())
            self.tree = self._parse_tree()
            self.sop = self._to_sop(self.tree, self.K)
            return True
        else:
            return False

def test():
    import time
    np.random.seed(42)
    gt_f = lambda x: (x[:,0] & x[:,1]) | (x[:,2] & x[:,3] & x[:,4])
    noise = 0.03
    X, y = gen_data(50, 30, gt_f, noise=noise)
    X_test, y_test = gen_data(100, 30, gt_f)
    #X, y = gen_data(42, 4, lambda x: (x[:,0] & x[:,1]) | x[:, 2])
    ws = 0.6 + 0.4 * np.random.binomial(1, 0.3, X.shape[0]) + np.random.randn(X.shape[0]) * 1e-4

    for nodes in range(10, 20):
        dt = SoftSATDT(nodes, X.shape[1], additional_constraints=True)
        #dt = SATDT(nodes, X.shape[1], additional_constraints=True)
        start = time.time()
        #if dt.fit(X, y, ws):
        if dt.fit(X, y):
            print(f'#### Nodes {nodes} ####')
            #print('sop:', dt.sop)
            print('tree:', dt.tree)
            print(f'error: {dt.error}')
            y_pred = dt.predict(X)
            train_acc = (y == y_pred).mean()

            y_test_pred = dt.predict(X_test)
            test_acc = (y_test == y_test_pred).mean()
            print(f'train acc: {train_acc:.3f}, test acc: {test_acc:.3f}')
        else:
            print(f'failed to solve for nodes = {nodes}')
        end = time.time()
        print(f'elapsed {end - start:0.3f} seconds.')

if __name__ == '__main__':
    test()
