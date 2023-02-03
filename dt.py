'''
Implements the DT->SAT algorithm from "Learning Optimal Decision Trees with SAT"
https://www.ijcai.org/proceedings/2018/0189.pdf
'''

import z3
from toydata import gen_data

def dt_sat_constraints(N, X):
    # initialize
    v = {}
    for i in range(1, N+1):
        v[i] = z3.Bool(f'v{i}')
    l = {}
    r = {}
    p = {}

    LR = [set(j for j in range(i+1, min(2 * i, N-1)+1) if (j % 2 == 0)) for i in range(N+1)]
    RR = [set(j for j in range(i+2, min(2 * i + 1, N)+1) if (j % 2 == 1)) for i in range(N+1)]
    for i in range(1, N+1):
        for j in LR[i]:
            l[i, j] = z3.Bool(f'l{i}{j}')
    for i in range(1, N+1):
        for j in RR[i]:
            r[i, j] = z3.Bool(f'r{i}{j}')
    for i in range(1, N):
        for j in range(2, N+1):
            p[j, i] = z3.Bool(f'p{j}{i}')
    
    print(v)
    print(l)
    print(r)
    print(p)
    assert False

def test():
    X, y = gen_data(20, 8, lambda x: (x[0] & x[1]) | (x[2] & x[3] & x[4]))

    constraints = dt_sat_constraints(5, X)


if __name__ == '__main__':
    test()
