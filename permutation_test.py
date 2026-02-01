import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

df = pd.read_csv('data/senate_primary_data.csv')

y = df['vote_share'].values
X_vars = ['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']
X = sm.add_constant(df[X_vars].copy())
groups = df['race_id'].values
unique_races = df['race_id'].unique()

model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': groups})
observed_t_endorse = model.tvalues['endorsement_score']
observed_t_fund = model.tvalues['ln_fundraising']

print(f"Observed endorsement beta = {model.params['endorsement_score']:.4f}, t = {observed_t_endorse:.4f}")
print(f"Observed fundraising beta = {model.params['ln_fundraising']:.4f}, t = {observed_t_fund:.4f}\n")

n_perms = 10000
perm_ts_endorse = np.zeros(n_perms)
perm_ts_fund = np.zeros(n_perms)

for i in range(n_perms):
    perm_endorse = df['endorsement_score'].values.copy()
    for race in unique_races:
        mask = groups == race
        vals = perm_endorse[mask]
        np.random.shuffle(vals)
        perm_endorse[mask] = vals

    X_perm = X.copy()
    X_perm['endorsement_score'] = perm_endorse
    perm_model = sm.OLS(y, X_perm).fit(cov_type='cluster', cov_kwds={'groups': groups})
    perm_ts_endorse[i] = perm_model.tvalues['endorsement_score']

perm_p_endorse = np.mean(np.abs(perm_ts_endorse) >= np.abs(observed_t_endorse))

for i in range(n_perms):
    perm_fund = df['ln_fundraising'].values.copy()
    for race in unique_races:
        mask = groups == race
        vals = perm_fund[mask]
        np.random.shuffle(vals)
        perm_fund[mask] = vals

    X_perm = X.copy()
    X_perm['ln_fundraising'] = perm_fund
    perm_model = sm.OLS(y, X_perm).fit(cov_type='cluster', cov_kwds={'groups': groups})
    perm_ts_fund[i] = perm_model.tvalues['ln_fundraising']

perm_p_fund = np.mean(np.abs(perm_ts_fund) >= np.abs(observed_t_fund))

print(f"Endorsement permutation p-value (two-sided, {n_perms:,} permutations): {perm_p_endorse:.4f}")
print(f"Fundraising permutation p-value (two-sided, {n_perms:,} permutations): {perm_p_fund:.4f}")
