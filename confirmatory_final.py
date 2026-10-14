"""Final analysis of the confirmatory generative study (60 cases x 6 conditions).
Pass 1 (quality, leakage): three independent raters (R1 teacher, R2 doctoral student, R3 doctoral student in CS).
Pass 2 (realised intensity) and pass 3 (action consistency): R1.
C5 rule: the self-check returned either VALID (draft kept) or a verdict with no revision; by the
pre-specified rule a verdict can never be the output, so the C4 draft is retained (C5 == C4 text)."""
import json, os, re, sys
import numpy as np, pandas as pd, krippendorff
from scipy import stats
from sklearn.metrics import cohen_kappa_score as K
BASE = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA = BASE + 'data_confirmatory/'
sys.path.insert(0, BASE)
from recal.stats_human import wilcoxon_rb, holm

D = ['concept_alignment_1_5', 'cognitive_alignment_1_5', 'knowledge_type_alignment_1_5',
     'pedagogical_appropriateness_1_5', 'correctness_1_5']
LK = 'leakage_0_1'
rng = np.random.default_rng(42); B = 5000
key = pd.read_csv(DATA+'KEY.csv')
cases = pd.read_json(BASE+'genkit/cases.jsonl', lines=True).set_index('case_id')
sheet = pd.read_csv(DATA+'outputs_blind.csv')[['blind_id', 'remediation', 'correct_answer']]
out = {}

# ---------- outputs ----------
T = sheet.merge(key, on='blind_id')
TT = T.pivot(index='case_id', columns='condition', values='remediation')
out['selfcheck'] = {'valid_kept': int((TT.C5 == TT.C4).sum()),
                    'verdict_without_revision': int(TT.C5.str.strip().str.upper().str.startswith('INVALID').sum())}
T['words'] = T.remediation.str.split().str.len()
out['median_words'] = T[~T.remediation.str.strip().str.upper().str.startswith('INVALID')].groupby('condition').words.median().to_dict()

def explicit(r):
    try: ans = json.loads(r.correct_answer)
    except Exception: return False
    if isinstance(ans, dict): ans = [x for v in ans.values() for x in v]
    return any(re.search(r'(answer|option|choice)\s*(is\s*)?\(?' + re.escape(str(a)) + r'\)?\b', str(r.remediation), re.I) for a in ans)
T['explicit'] = T.apply(explicit, axis=1)
EX = T.pivot(index='case_id', columns='condition', values='explicit').astype(int)
EX['C5'] = EX['C4']
out['explicit_leak'] = EX.sum().to_dict()

# ---------- pass 1 ----------
R = {'R1': pd.read_csv(DATA+'pass1_R1.csv'), 'R2': pd.read_csv(DATA+'pass1_R2.csv'),
     'R3': pd.read_csv(DATA+'pass1_R3.csv')}
L = pd.concat([r.merge(key, on='blind_id').assign(rater=k) for k, r in R.items()], ignore_index=True)
c4 = L[L.condition == 'C4'].set_index(['rater', 'case_id'])
m = L.condition == 'C5'
for c in D + [LK]:
    L.loc[m, c] = [c4.at[(r, cid), c] for r, cid in zip(L.loc[m, 'rater'], L.loc[m, 'case_id'])]
L['Q'] = L[D].mean(1)
X = L[L.condition != 'C5']
out['alpha'] = {c: krippendorff.alpha(reliability_data=X.pivot_table(index='rater', columns='blind_id', values=c).values,
                                      level_of_measurement='interval' if c == 'Q' else 'ordinal') for c in D + ['Q']}
out['alpha_leak_R1R2'] = krippendorff.alpha(reliability_data=X[X.rater.isin(['R1', 'R2'])].pivot_table(index='rater', columns='blind_id', values=LK).values, level_of_measurement='nominal')
pw = {}
for a, b in [('R1', 'R2'), ('R1', 'R3'), ('R2', 'R3')]:
    A = X[X.rater == a].set_index('blind_id').sort_index(); Bq = X[X.rater == b].set_index('blind_id').sort_index()
    pw[f'{a}-{b}'] = K(np.concatenate([A[c] for c in D]), np.concatenate([Bq[c] for c in D]), weights='quadratic')
out['pairwise_qwk'] = pw
conds = ['C3', 'C4', 'C4M', 'C4H', 'C4X']
Qb = L.groupby(['case_id', 'condition']).Q.mean().unstack()
desc = []
for c in conds:
    v = Qb[c].to_numpy(); bs = [rng.choice(v, len(v)).mean() for _ in range(B)]
    desc.append({'condition': c, 'mean': v.mean(), 'sd': v.std(ddof=1), 'lo': np.percentile(bs, 2.5), 'hi': np.percentile(bs, 97.5)})
out['quality'] = pd.DataFrame(desc)
fr = stats.friedmanchisquare(*[Qb[c] for c in conds])
out['friedman'] = {'chi2': fr.statistic, 'df': len(conds) - 1, 'p': fr.pvalue, 'W': fr.statistic / (len(Qb) * (len(conds) - 1))}
rows = [{'contrast': f'{x} vs {y}', **wilcoxon_rb(Qb[x], Qb[y])} for x, y in [('C4M', 'C4'), ('C4H', 'C4'), ('C3', 'C4'), ('C4X', 'C4')]]
for r, p in zip(rows[:2], holm([r['p'] for r in rows[:2]])): r['p_holm'] = p
out['quality_contrasts'] = pd.DataFrame(rows)
out['dims'] = L[L.condition.isin(conds)].groupby('condition')[D].mean().loc[conds]
out['quality_by_rater'] = L.groupby(['rater', 'condition']).Q.mean().unstack()[conds]
# leakage: flagged by R1 or R2 (R3 excluded: did not apply the rubric definition, flagged 2/10 explicit disclosures)
Lk = L[L.rater.isin(['R1', 'R2'])].groupby(['case_id', 'condition'])[LK].max().unstack()
out['leak_any'] = Lk[conds].mean().to_dict()
out['leak_by_rater'] = L.groupby(['rater', 'condition'])[LK].mean().unstack()[conds]
from statsmodels.stats.contingency_tables import mcnemar
def mc(a, b):
    t = pd.crosstab(Lk[a], Lk[b]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    return {'n01': int(t.loc[0, 1]), 'n10': int(t.loc[1, 0]), 'p': mcnemar(t.values, exact=True).pvalue}
out['leak_tests'] = {'C3 vs C4': mc('C3', 'C4'), 'C4M vs C4': mc('C4M', 'C4'), 'C4H vs C4': mc('C4H', 'C4')}
out['explicit_test_C3_C4'] = mcnemar(pd.crosstab(EX.C3, EX.C4).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values, exact=True).pvalue
expl_flag = T[T.explicit].merge(L[L.rater.isin(['R1', 'R2', 'R3'])][['blind_id', 'rater', LK]], on='blind_id').groupby('rater')[LK].mean()
out['explicit_detected_by_rater'] = expl_flag.to_dict()

# ---------- pass 2 / 3 (R1) ----------
p2 = pd.read_csv(DATA+'pass2_R1.csv').merge(key, on='blind_id')
p3 = pd.read_csv(DATA+'pass3_R1.csv').merge(key, on='blind_id')
I = p2.pivot(index='case_id', columns='condition', values='intensity_1_5'); I['C5'] = I['C4']
pt = cases.loc[I.index, 'p_t']; reg = cases.loc[I.index, 'region_C']
out['rho'] = {c: stats.spearmanr(pt, I[c]).statistic for c in conds}
out['rho_p'] = {c: stats.spearmanr(pt, I[c]).pvalue for c in conds}
idx = I.index.to_numpy(); diffs = {'C4-C3': [], 'C4-C4M': [], 'C4-C4H': []}
for _ in range(B):
    s = rng.choice(idx, len(idx))
    r = {c: stats.spearmanr(pt.loc[s], I.loc[s, c]).statistic for c in ['C3', 'C4', 'C4M', 'C4H']}
    diffs['C4-C3'].append(r['C4'] - r['C3']); diffs['C4-C4M'].append(r['C4'] - r['C4M']); diffs['C4-C4H'].append(r['C4'] - r['C4H'])
out['rho_diff_ci'] = {k: (np.mean(v), *np.nanpercentile(v, [2.5, 97.5])) for k, v in diffs.items()}
# objective complementary measure (post hoc): output length in words as a proxy of the amount of help
Wd = T.assign(words=T.remediation.str.split().str.len()).pivot(index='case_id', columns='condition', values='words')
Wd['C5'] = Wd['C4']; ptw = cases.loc[Wd.index, 'p_t']
out['rho_words'] = {c: stats.spearmanr(ptw, Wd[c]).statistic for c in conds}
dw = {'C4-C3': [], 'C4-C4M': [], 'C4-C4H': []}; idw = Wd.index.to_numpy()
for _ in range(B):
    s = rng.choice(idw, len(idw)); r4 = stats.spearmanr(ptw.loc[s], Wd.loc[s, 'C4']).statistic
    for k in dw: dw[k].append(r4 - stats.spearmanr(ptw.loc[s], Wd.loc[s, k.split('-')[1]]).statistic)
out['rho_words_diff_ci'] = {k: (out['rho_words']['C4'] - out['rho_words'][k.split('-')[1]], *np.percentile(v, [2.5, 97.5])) for k, v in dw.items()}
out['words_by_region'] = Wd[conds].groupby(cases.loc[Wd.index, 'region_C']).median().loc[['LOW', 'MEDIUM', 'HIGH']]
out['rho_intensity_words'] = stats.spearmanr(p2.merge(T[['blind_id', 'words']], on='blind_id').query('condition != "C5"')[['intensity_1_5', 'words']]).statistic
out['intensity_by_region'] = I[conds].groupby(reg).mean().loc[['LOW', 'MEDIUM', 'HIGH']]
A = p3.pivot(index='case_id', columns='condition', values='action_consistency_1_5'); A['C5'] = A['C4']
out['action'] = A[['C4', 'C4M', 'C4H', 'C4X']].mean().to_dict()
a3 = p3[p3.condition == 'C4'].merge(cases[['region_C']], left_on='case_id', right_index=True)
out['action_by_mandated'] = p3[p3.condition == 'C4'].groupby('action').action_consistency_1_5.agg(['mean', 'count'])
out['quality_gain_by_region'] = (Qb['C4'] - Qb['C3']).groupby(cases.loc[Qb.index, 'region_C']).agg(['mean', 'std', 'count'])

pd.set_option('display.width', 220); pd.set_option('display.max_columns', 20)
with open(BASE+'results/confirmatory_final.txt', 'w') as f:
    for k, v in out.items():
        s = f'== {k}\n{v}\n'; print(s); f.write(s)
L.to_csv(BASE+'results/confirmatory_pass1_long.csv', index=False)
