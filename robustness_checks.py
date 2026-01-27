import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('data/senate_primary_data.csv')
df['trump_era'] = (df['year'] >= 2016).astype(int)

y = df['vote_share']

X = sm.add_constant(df[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
m1 = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})
print(f"Main Model (N={len(df)}): endorsement β={m1.params['endorsement_score']:.2f}, p={m1.pvalues['endorsement_score']:.3f}")

X = sm.add_constant(df[['ln_full_cycle', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
m2 = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})
print(f"Full Cycle (N={len(df)}): endorsement β={m2.params['endorsement_score']:.2f}, p={m2.pvalues['endorsement_score']:.3f}")

df_pre = df[df['trump_era'] == 0]
X = sm.add_constant(df_pre[['ln_fundraising', 'endorsement_score', 'prior_office', 'field_size']])
m3 = sm.OLS(df_pre['vote_share'], X).fit(cov_type='cluster', cov_kwds={'groups': df_pre['race_id']})
print(f"Pre-Trump (N={len(df_pre)}): endorsement β={m3.params['endorsement_score']:.2f}, p={m3.pvalues['endorsement_score']:.3f}")

df_trump = df[df['trump_era'] == 1]
X = sm.add_constant(df_trump[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
m4 = sm.OLS(df_trump['vote_share'], X).fit(cov_type='cluster', cov_kwds={'groups': df_trump['race_id']})
print(f"Trump Era (N={len(df_trump)}): endorsement β={m4.params['endorsement_score']:.2f}, p={m4.pvalues['endorsement_score']:.3f}")

df_open = df[df['race_type'] == 'open_seat']
X = sm.add_constant(df_open[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
m5 = sm.OLS(df_open['vote_share'], X).fit(cov_type='cluster', cov_kwds={'groups': df_open['race_id']})
print(f"Open-Seat (N={len(df_open)}): endorsement β={m5.params['endorsement_score']:.2f}, p={m5.pvalues['endorsement_score']:.3f}")

df_inc = df[df['race_type'] == 'incumbent_challenge']
X = sm.add_constant(df_inc[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
m6 = sm.OLS(df_inc['vote_share'], X).fit(cov_type='cluster', cov_kwds={'groups': df_inc['race_id']})
print(f"Incumbent (N={len(df_inc)}): endorsement β={m6.params['endorsement_score']:.2f}, p={m6.pvalues['endorsement_score']:.3f}")

df_pac = df[df['ln_super_pac'].notna() & (df['ln_super_pac'] > 0)]
if len(df_pac) >= 10:
    X = sm.add_constant(df_pac[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size', 'ln_super_pac']])
    m7 = sm.OLS(df_pac['vote_share'], X).fit(cov_type='cluster', cov_kwds={'groups': df_pac['race_id']})
    print(f"Super PAC (N={len(df_pac)}): endorsement β={m7.params['endorsement_score']:.2f}, p={m7.pvalues['endorsement_score']:.3f}")
