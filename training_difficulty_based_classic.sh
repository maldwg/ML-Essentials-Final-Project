#!/bin/sh

AGENT=baby_terminator
# Different number of rounds to learn the game incrementally on each scenario
# larger nr of round vs weak agents to learn the scenario deeper
N_ROUNDS=1000

rm agent_code/$AGENT/logs/all.log
touch agent_code/$AGENT/logs/all.log

 for STAGE in "peaceful_agent peaceful_agent random_agent" "peaceful_agent random_agent random_agent" "peaceful_agent random_agent rule_based_agent" " rule_based_agent rule_based_agent rule_based_agent"
do
    echo "training on stage $STAGE"
    python main.py play --agents "$AGENT" $STAGE --n-rounds="$N_ROUNDS" --train 1  --no-gui --scenario "classic" 
    python evaluations_$AGENT.py
    cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
done