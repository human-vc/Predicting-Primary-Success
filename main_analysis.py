import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('data/senate_primary_data.csv')

y = df['vote_share']

X1 = sm.add_constant(df[['ln_fundraising']])
model1 = sm.OLS(y, X1).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})

X2 = sm.add_constant(df[['endorsement_score']])
model2 = sm.OLS(y, X2).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})

X3 = sm.add_constant(df[['ln_fundraising', 'endorsement_score']])
model3 = sm.OLS(y, X3).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})

X4 = sm.add_constant(df[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
model4 = sm.OLS(y, X4).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})

df['traditional_endorsement'] = df['endorsement_score'] - (df['trump_endorsement'] * 6)
X5 = sm.add_constant(df[['ln_fundraising', 'traditional_endorsement', 'trump_endorsement', 'prior_office', 'field_size']])
model5 = sm.OLS(y, X5).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})

print(model4.summary())
print(model5.summary())

trump_endorsed = df[df['trump_endorsement'] == 1]
non_trump = df[df['trump_endorsement'] == 0]

print(f"Trump-endorsed: n={len(trump_endorsed)}, mean={trump_endorsed['vote_share'].mean():.1f}%, win rate={trump_endorsed['winner'].mean()*100:.0f}%")
print(f"Non-Trump: n={len(non_trump)}, mean={non_trump['vote_share'].mean():.1f}%, win rate={non_trump['winner'].mean()*100:.0f}%")

X_vif = sm.add_constant(df[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
for i, col in enumerate(X_vif.columns):
    if col != 'const':
        print(f"VIF {col}: {variance_inflation_factor(X_vif.values, i):.2f}")

np.random.seed(42)
n_bootstrap = 5000
boot_coefs = {'endorsement': [], 'fundraising': [], 'trump': []}

for _ in range(n_bootstrap):
    boot_idx = np.random.choice(len(df), size=len(df), replace=True)
    boot_df = df.iloc[boot_idx]
    y_boot = boot_df['vote_share']
    X_boot = sm.add_constant(boot_df[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
    try:
        model_boot = sm.OLS(y_boot, X_boot).fit()
        boot_coefs['endorsement'].append(model_boot.params['endorsement_score'])
        boot_coefs['fundraising'].append(model_boot.params['ln_fundraising'])
        boot_coefs['trump'].append(model_boot.params['trump_endorsement'])
    except:
        continue

for var in ['endorsement', 'fundraising', 'trump']:
    ci_lower = np.percentile(boot_coefs[var], 2.5)
    ci_upper = np.percentile(boot_coefs[var], 97.5)
    print(f"{var}: 95% CI [{ci_lower:.2f}, {ci_upper:.2f}]")
