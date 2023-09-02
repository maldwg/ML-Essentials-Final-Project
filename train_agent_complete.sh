#!/bin/sh

AGENT=baby_terminator
# Different number of rounds to learn the game incrementally on each scenario
# larger nr of round vs weak agents to learn the scenario deeper
N_ROUNDS_1=75
N_ROUNDS_2=50
N_ROUNDS_3=30
N_ROUNDS_4=30

rm agent_code/$AGENT/logs/all.log
touch agent_code/$AGENT/logs/all.log

for SCENARIO in "empty" "coin-heaven" # "loot-crate" "classic"
do
    echo "training on scenario $SCENARIO"
    echo "Training step 1 / 4"
    python main.py play --agents "$AGENT" peaceful_agent peaceful_agent random_agent --n-rounds="$N_ROUNDS_1" --train 1  --no-gui --scenario "$SCENARIO" 
    python evaluations.py 
    cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
    echo "Training step 2 / 4"
    python main.py play --agents "$AGENT" peaceful_agent random_agent random_agent --n-rounds="$N_ROUNDS_2" --train 1  --no-gui --scenario "$SCENARIO" 
    python evaluations.py
    cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
    echo "Training step 3 / 4"
    python main.py play --agents "$AGENT"  peaceful_agent random_agent rule_based_agent --n-rounds="$N_ROUNDS_3" --train 1  --no-gui --scenario "$SCENARIO" 
    python evaluations.py
    cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
    echo "Training step 4 / 4"
    python main.py play --agents "$AGENT" rule_based_agent rule_based_agent rule_based_agent --n-rounds="$N_ROUNDS_4" --train 1  --no-gui --scenario "$SCENARIO" 
    python evaluations.py
    cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
done