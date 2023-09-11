#!/bin/bash

AGENT=baby_terminator
# Different number of rounds to learn the game incrementally on each scenario
# larger nr of round vs weak agents to learn the scenario deeper
CHECKPOINT=250
currentdatetime=$(date +"%Y%m%d%H%M")


rm agent_code/$AGENT/logs/all.log
touch agent_code/$AGENT/logs/all.log

PARENT_DIR=agent_code/$AGENT/checkpoints/$currentdatetime
mkdir -p $PARENT_DIR


for idx in {1..40}
do
    rounds=$(( CHECKPOINT * idx ))
    echo "new upper bound is ${rounds} rounds"
    mkdir -p  $PARENT_DIR/$rounds
    echo "Train the agent... ))"
    python main.py play --agents baby_terminator --n-rounds=$CHECKPOINT --train 1 --scenario loot-crate --no-gui
    pid=$(pgrep -f "python main.py play")
    tail --pid="$pid" -f /dev/null
    echo "Training finished"
    echo "copy the old model"
    cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$rounds/
    echo "evaluate the agent"
    python evaluations.py
    cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
    echo "---------------------------------------------------"
done