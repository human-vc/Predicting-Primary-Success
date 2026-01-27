import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

df = pd.read_csv('data/senate_primary_data.csv')
df['trump_era'] = (df['year'] >= 2016).astype(int)

y = df['vote_share']
X = sm.add_constant(df[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})

fig, ax = plt.subplots(figsize=(8, 5))
vars_plot = ['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']
labels = ['Log(Q1 Fundraising)', 'Endorsement Score', 'Trump Endorsement', 'Prior Office', 'Field Size']
coefs = [model.params[v] for v in vars_plot]
ses = [model.bse[v] for v in vars_plot]
pvals = [model.pvalues[v] for v in vars_plot]
y_pos = np.arange(len(vars_plot))
for i, (coef, se, p) in enumerate(zip(coefs, ses, pvals)):
    color = '#1a5276' if p < 0.05 else '#e67e22'
    ax.plot([coef - 1.96*se, coef + 1.96*se], [i, i], color=color, linewidth=3, solid_capstyle='round')
    ax.scatter(coef, i, s=120, c=color, edgecolors='white', linewidth=1.5, zorder=5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel('Coefficient Estimate (95% CI)', fontsize=11)
ax.set_xlim(-12, 12)
plt.tight_layout()
plt.savefig('figures/fig1_coefficients.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figures/fig1_coefficients.pdf', bbox_inches='tight', facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
endorse_range = np.linspace(0, 20, 100)
mean_ln_fund = df['ln_fundraising'].mean()
mean_prior = df['prior_office'].mean()
mean_field = df['field_size'].mean()
pred_no_trump, pred_trump, se_no_trump, se_trump = [], [], [], []
for e in endorse_range:
    X_pred = pd.DataFrame({'const': [1], 'ln_fundraising': [mean_ln_fund], 'endorsement_score': [e],
                           'trump_endorsement': [0], 'prior_office': [mean_prior], 'field_size': [mean_field]})
    pred = model.get_prediction(X_pred)
    pred_no_trump.append(pred.predicted_mean[0])
    se_no_trump.append(pred.se_mean[0])
    X_pred['trump_endorsement'] = [1]
    pred = model.get_prediction(X_pred)
    pred_trump.append(pred.predicted_mean[0])
    se_trump.append(pred.se_mean[0])
pred_no_trump, pred_trump = np.array(pred_no_trump), np.array(pred_trump)
se_no_trump, se_trump = np.array(se_no_trump), np.array(se_trump)
ax.fill_between(endorse_range, pred_no_trump - 1.96*se_no_trump, pred_no_trump + 1.96*se_no_trump, alpha=0.2, color='#2980b9')
ax.plot(endorse_range, pred_no_trump, color='#2980b9', linewidth=2.5)
ax.fill_between(endorse_range, pred_trump - 1.96*se_trump, pred_trump + 1.96*se_trump, alpha=0.2, color='#c0392b')
ax.plot(endorse_range, pred_trump, color='#c0392b', linewidth=2.5)
non_trump = df[df['trump_endorsement'] == 0]
trump = df[df['trump_endorsement'] == 1]
ax.scatter(non_trump['endorsement_score'], non_trump['vote_share'], s=50, c='#2980b9', alpha=0.5, edgecolors='white', linewidth=0.5)
ax.scatter(trump['endorsement_score'], trump['vote_share'], s=100, c='#c0392b', alpha=0.8, edgecolors='white', linewidth=1, marker='s')
ax.set_xlabel('Endorsement Score', fontsize=11)
ax.set_ylabel('Predicted Vote Share (%)', fontsize=11)
ax.set_xlim(-0.5, 20)
ax.set_ylim(0, 70)
plt.tight_layout()
plt.savefig('figures/fig2_marginal_effects.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figures/fig2_marginal_effects.pdf', bbox_inches='tight', facecolor='white')
plt.close()

specs = {}
X = sm.add_constant(df[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
m = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})
specs['Main Model\n(N=45)'] = (m.params['endorsement_score'], m.bse['endorsement_score'], m.pvalues['endorsement_score'])
X = sm.add_constant(df[['ln_full_cycle', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
m = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})
specs['Full Cycle\n(N=45)'] = (m.params['endorsement_score'], m.bse['endorsement_score'], m.pvalues['endorsement_score'])
df_pre = df[df['trump_era'] == 0]
X = sm.add_constant(df_pre[['ln_fundraising', 'endorsement_score', 'prior_office', 'field_size']])
m = sm.OLS(df_pre['vote_share'], X).fit(cov_type='cluster', cov_kwds={'groups': df_pre['race_id']})
specs['Pre-Trump\n(N=26)'] = (m.params['endorsement_score'], m.bse['endorsement_score'], m.pvalues['endorsement_score'])
df_trump = df[df['trump_era'] == 1]
X = sm.add_constant(df_trump[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
m = sm.OLS(df_trump['vote_share'], X).fit(cov_type='cluster', cov_kwds={'groups': df_trump['race_id']})
specs['Trump Era\n(N=19)'] = (m.params['endorsement_score'], m.bse['endorsement_score'], m.pvalues['endorsement_score'])
df_open = df[df['race_type'] == 'open_seat']
X = sm.add_constant(df_open[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
m = sm.OLS(df_open['vote_share'], X).fit(cov_type='cluster', cov_kwds={'groups': df_open['race_id']})
specs['Open-Seat\n(N=31)'] = (m.params['endorsement_score'], m.bse['endorsement_score'], m.pvalues['endorsement_score'])
df_inc = df[df['race_type'] == 'incumbent_challenge']
X = sm.add_constant(df_inc[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
m = sm.OLS(df_inc['vote_share'], X).fit(cov_type='cluster', cov_kwds={'groups': df_inc['race_id']})
specs['Incumbent\n(N=14)'] = (m.params['endorsement_score'], m.bse['endorsement_score'], m.pvalues['endorsement_score'])
fig, ax = plt.subplots(figsize=(9, 5))
labels = list(specs.keys())
coefs = [specs[l][0] for l in labels]
ses = [specs[l][1] for l in labels]
pvals = [specs[l][2] for l in labels]
x_pos = np.arange(len(labels))
for i, (coef, se, p) in enumerate(zip(coefs, ses, pvals)):
    color = '#1a5276' if p < 0.05 else '#e67e22'
    ax.plot([i, i], [coef - 1.96*se, coef + 1.96*se], color=color, linewidth=3)
    ax.scatter(i, coef, s=140, c=color, edgecolors='white', linewidth=1.5, zorder=5)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Endorsement Score Coefficient (95% CI)', fontsize=11)
ax.set_ylim(-0.5, 2.5)
plt.tight_layout()
plt.savefig('figures/fig3_robustness.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figures/fig3_robustness.pdf', bbox_inches='tight', facecolor='white')
plt.close()
