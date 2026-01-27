import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data/senate_primary_data.csv')

def win_prediction():
    y = df['winner']
    X = sm.add_constant(df[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
    model = sm.Logit(y, X).fit(disp=0)
    odds_ratios = np.exp(model.params)
    y_pred = model.predict(X)
    return {
        'model': model,
        'odds_ratios': odds_ratios,
        'auc': roc_auc_score(y, y_pred),
        'accuracy': ((y_pred > 0.5).astype(int) == y).mean()
    }

def counterfactuals():
    y = df['vote_share']
    X = sm.add_constant(df[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 'prior_office', 'field_size']])
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})
    results = []
    for _, row in df[df['trump_endorsement'] == 1].iterrows():
        X_with = pd.DataFrame({'const': [1], 'ln_fundraising': [row['ln_fundraising']], 
            'endorsement_score': [row['endorsement_score']], 'trump_endorsement': [1],
            'prior_office': [row['prior_office']], 'field_size': [row['field_size']]})
        X_without = X_with.copy()
        X_without['trump_endorsement'] = 0
        X_without['endorsement_score'] = row['endorsement_score'] - 6
        results.append({
            'candidate': row['candidate_name'], 'actual': row['vote_share'],
            'counterfactual': model.predict(X_without)[0]
        })
    return results

def trump_field_interaction():
    df['trump_x_field'] = df['trump_endorsement'] * df['field_size']
    y = df['vote_share']
    X = sm.add_constant(df[['ln_fundraising', 'endorsement_score', 'trump_endorsement', 
        'prior_office', 'field_size', 'trump_x_field']])
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})
    return {n: model.params['trump_endorsement'] + model.params['trump_x_field'] * n for n in [2,3,4,5]}

def endorsement_thresholds():
    winners = df[df['winner'] == 1]['endorsement_score']
    losers = df[df['winner'] == 0]['endorsement_score']
    return {
        'winner_mean': winners.mean(),
        'loser_mean': losers.mean(),
        't_stat': stats.ttest_ind(winners, losers)[0],
        'p_val': stats.ttest_ind(winners, losers)[1]
    }

def out_of_state_placebo():
    oos = {'Todd Tiahrt': 1, 'Kelly Ayotte': 1, 'Ted Cruz': 10, 'Richard Mourdock': 1, 
           'Chris McDaniel': 1, 'Thad Cochran': 3, 'Mo Brooks': 6, 'Manny Sethi': 6}
    df['out_of_state'] = df['candidate_name'].map(oos).fillna(0)
    df['in_state'] = df['endorsement_score'] - df['out_of_state']
    y = df['vote_share']
    X = sm.add_constant(df[['ln_fundraising', 'in_state', 'out_of_state', 'trump_endorsement', 'prior_office', 'field_size']])
    return sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['race_id']})

if __name__ == '__main__':
    wp = win_prediction()
    print(f"Win Prediction: Endorsement OR={wp['odds_ratios']['endorsement_score']:.2f}, AUC={wp['auc']:.3f}")
    print(f"\nCounterfactuals:")
    for c in counterfactuals():
        print(f"  {c['candidate']}: {c['actual']:.1f}% -> {c['counterfactual']:.1f}%")
    print(f"\nTrump x Field: {trump_field_interaction()}")
    print(f"\nThresholds: {endorsement_thresholds()}")
    m = out_of_state_placebo()
    print(f"\nPlacebo: In-state p={m.pvalues['in_state']:.3f}, Out-of-state p={m.pvalues['out_of_state']:.3f}")
