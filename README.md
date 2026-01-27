# Predicting Primary Success

Replication materials for "The Party Decides in Senate Primaries: Endorsements, Fundraising, and the Trump Effect, 2008-2022"

## Overview

This repository contains data and code for analyzing the determinants of vote share in Republican Senate primaries. We test competing predictions from candidate-centered models (emphasizing fundraising) and the "Party Decides" framework (emphasizing elite endorsements).

Key findings:
- Endorsement score significantly predicts vote share (β = 0.67, p = 0.003)
- Early fundraising shows no independent effect once endorsements are controlled (β = 1.12, p = 0.564)
- Trump endorsements carry exceptional influence (β = 12.1, p = 0.006), with all four Trump-endorsed candidates winning their primaries
- Results are robust across seven specifications including temporal splits, race type stratification, and alternative fundraising measures

## Data

senate_primary_data.csv contains 45 candidates across 16 competitive Republican Senate primaries (2008-2022) in 12 states.

Variables: race_id, year, state, race_type, candidate_name, vote_share, winner, q1_fundraising, ln_fundraising, endorsement_score, trump_endorsement, prior_office, field_size, full_cycle_receipts, ln_full_cycle, ln_super_pac

Endorsement weighting: U.S. Senator (3 points), same-state Governor (6 points), same-state U.S. Representative (2 points), Trump (6 points)

## Code

main_analysis.py: Primary OLS regression models, Trump endorsement comparison, variance inflation factors, bootstrap confidence intervals

robustness_checks.py: Seven robustness specifications including full cycle fundraising, temporal splits, race type stratification, Super PAC controls

figures.py: Publication-quality coefficient plot, marginal effects plot, robustness plot

## Requirements

pandas>=2.1.0, numpy>=1.24.0, scipy>=1.11.0, statsmodels>=0.14.0, matplotlib>=3.8.0

## Data Sources

Endorsements: FiveThirtyEight endorsement tracker, Ballotpedia, campaign press releases, newspaper archives

Fundraising: Federal Election Commission filings

Vote share: State election authority certified results

## Citation

Crainic, J. (2025). The Party Decides in Senate Primaries: Endorsements, Fundraising, and the Trump Effect, 2008-2022. Working Paper.

## License

MIT License
