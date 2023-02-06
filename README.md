# z3-dt

Implements the DT->SAT algorithm from "Learning Optimal Decision Trees with SAT"
https://www.ijcai.org/proceedings/2018/0189.pdf

## Example usage

```python
import numpy as np
from dt import SoftSATDT, SATDT
from toydata import gen_data

np.random.seed(42)
gt_f = lambda x: (x[:,0] & x[:,1]) | (x[:,2] & x[:,3] & x[:,4])
X, y = gen_data(50, 10, gt_f, noise=0.05)
X_test, y_test = gen_data(100, 10, gt_f)

for nodes in range(10, 20):
    dt = SoftSATDT(nodes, X.shape[1])
    if dt.fit(X, y):
        print(dt.tree)
        y_pred = dt.predict(X)
        train_acc = (y == y_pred).mean()

        y_test_pred = dt.predict(X_test)
        test_acc = (y_test == y_test_pred).mean()
        print(f'Nodes {nodes} error: {dt.error}')
        print(f'train acc: {train_acc:.3f}, test acc: {test_acc:.3f}')
    else:
        print(f'failed to solve for nodes = {nodes}')
```
