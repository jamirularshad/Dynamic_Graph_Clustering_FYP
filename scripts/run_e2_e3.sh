#!/usr/bin/env bash
# E2/E3 pilot grid. Usage:
#   scripts/run_e2_e3.sh tune            # EWMA gamma on validation seeds 101-103
#   scripts/run_e2_e3.sh select GAMMA    # causal estimators on validation seeds 101-103
#   scripts/run_e2_e3.sh test GAMMA      # all methods on test seeds 1-3
set -euo pipefail
PY=.venv/bin/python
RUN() { $PY -m dgc.runner.run "$@" --out results >/dev/null; }

if [ "$1" = tune ]; then
  for regime in gradual abrupt; do for seed in 101 102 103; do for g in 0.2 0.5 0.8; do
    RUN configs/e2e3_$regime.yaml --graph-seed $seed --algo-seed $seed \
      --param scorer=ewma --param ewma_gamma=$g --name tune_${regime}_ewma_g$g
  done; done; done
  exit 0
fi

GAMMA=$2
if [ "$1" = select ]; then
  for regime in gradual abrupt; do for seed in 101 102 103; do
    for scorer in eq14 one_step ewma profile combined; do
      RUN configs/e2e3_$regime.yaml --graph-seed $seed --algo-seed $seed \
        --param scorer=$scorer --param ewma_gamma=$GAMMA --name select_${regime}_$scorer
    done
  done; done
  exit 0
fi

for regime in gradual abrupt; do for seed in 1 2 3; do
  S="--graph-seed $seed --algo-seed $seed"
  C=configs/e2e3_$regime.yaml
  RUN $C $S --method dygmf_lookahead --param scorer=eq14 --name e2e3_${regime}_lookahead_eq14
  for scorer in eq14 one_step profile combined random; do
    RUN $C $S --param scorer=$scorer --param ewma_gamma=$GAMMA --name e2e3_${regime}_causal_$scorer
  done
  RUN $C $S --param scorer=ewma --param ewma_gamma=$GAMMA --name e2e3_${regime}_causal_ewma
  RUN $C $S --param mu=1.0 --name e2e3_${regime}_full_update
  RUN $C $S --method louvain --clear-params --name e2e3_${regime}_louvain
done; done