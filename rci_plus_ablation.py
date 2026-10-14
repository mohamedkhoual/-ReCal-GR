"""Post-hoc ablation (Section 5.2): does a history modifier or an uncertainty band improve the RCI-C rule?
Input: results/cohort_scored.pkl (written by run_all.py) and results/revision/episodes_folds_oof.csv."""
import sys
import numpy as np, pandas as pd
import statsmodels.api as sm
c = pd.read_pickle(sys.argv[1] if len(sys.argv) > 1 else 'results/cohort_scored.pkl').reset_index(drop=True)
o = pd.read_csv('results/revision/episodes_folds_oof.csv')
assert np.allclose(c.p_t.values, o.p_logreg_oof.values)
c['fold'] = o.fold.values
O, U = np.array([0, 0.25, 1.25]), np.array([2, 1, 0])
reg = lambda p, t1=.5, t2=.8: np.where(p <= t1, 2, np.where(p <= t2, 1, 0))
loss = lambda k, y: np.where(y == 1, O[k], U[k])
y, p = c.y.values, c.p_t.values
h = (c.recent_failure_ratio.fillna(c.recent_failure_ratio.mean()) + np.minimum(c.consecutive_failures.fillna(0), 3) / 3) / 2
h = (h - h.mean()).values
print('RCI-C risk %.4f' % loss(reg(p), y).mean())
lams = np.round(np.arange(-0.3, 0.31, 0.02), 2); held = []; chosen = []
for f in sorted(c.fold.unique()):          # lambda chosen on training learners only
    tr = (c.fold != f).values
    best = min(lams, key=lambda l: loss(reg(p[tr] - l * h[tr]), y[tr]).mean())
    chosen.append(best); held.append(loss(reg(p[~tr] - best * h[~tr]), y[~tr]))
print('history modifier: lambda per fold', chosen, 'held-out risk %.4f' % np.concatenate(held).mean())
for d in [0, .02, .05, .10]:
    k = reg(p); k = np.where((np.abs(p - .5) < d) | (np.abs(p - .8) < d), 1, k)
    print('uncertainty band +/-%.2f: risk %.4f, over-assist %.3f, under-assist %.3f' % (d, loss(k, y).mean(), np.mean(k[y == 1] == 2), np.mean(k[y == 0] == 0)))
m = sm.Logit(y, sm.add_constant(np.column_stack([np.log(p / (1 - p)), h]))).fit(disp=0)
print('incremental value of history given logit(p): coef %.3f, p = %.2f' % (m.params[2], m.pvalues[2]))
