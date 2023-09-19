#!/bin/bash

AGENT=baby_terminator
# Different number of rounds to learn the game incrementally on each scenario
# larger nr of round vs weak agents to learn the scenario deeper
CHECKPOINT=100

rm agent_code/$AGENT/logs/all.log
touch agent_code/$AGENT/logs/all.log

currentdatetime=$(date +"%Y%m%d%H%M")
PARENT_DIR=agent_code/$AGENT/checkpoints/$currentdatetime
mkdir -p $PARENT_DIR

# currentdatetime=$(date +"%Y%m%d%H%M")
# for idx in {1..3}
# do
#     rounds=$(( CHECKPOINT * idx ))
#     echo "new upper bound is ${rounds} rounds"
#     mkdir -p  $PARENT_DIR/$currentdatetime/$rounds
#     echo "Train the agent... ))"
#     python main.py play --agents baby_terminator --n-rounds=$CHECKPOINT --train 1 --scenario coin-heaven --no-gui
#     sleep 30
#     echo "Training finished"
#     echo "copy the old model"
#     cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$currentdatetime/$rounds/
#     echo "evaluate the agent"
#     python evaluations_$AGENT.py
#     cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
#     echo "---------------------------------------------------"
#     rounds=$(( CHECKPOINT * idx + CHECKPOINT ))
#     echo "new upper bound is ${rounds} rounds"
#     mkdir -p  $PARENT_DIR/$currentdatetime/$rounds
#     echo "Train the agent... ))"
#     python main.py play --agents baby_terminator --n-rounds=$CHECKPOINT --train 1 --scenario loot-crate --no-gui
#     sleep 30
#     echo "Training finished"
#     echo "copy the old model"
#     cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$currentdatetime/$rounds/
#     echo "evaluate the agent"
#     python evaluations_$AGENT.py
#     cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
#     echo "---------------------------------------------------"
#     rounds=$(( CHECKPOINT * idx + 2 * CHECKPOINT))
#     echo "new upper bound is ${rounds} rounds"
#     mkdir -p  $PARENT_DIR/$currentdatetime/$rounds
#     echo "Train the agent... ))"
#     python main.py play --agents baby_terminator --n-rounds=$CHECKPOINT --train 1 --scenario classic --no-gui
#     sleep 30
#     echo "Training finished"
#     echo "copy the old model"
#     cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$currentdatetime/$rounds/
#     echo "evaluate the agent"
#     python evaluations_$AGENT.py
#     cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
#     echo "---------------------------------------------------"
# done

currentdatetime=$(date +"%Y%m%d%H%M")
for idx in {1..10}
do
    rounds=$(( CHECKPOINT * idx ))
    echo "new upper bound is ${rounds} rounds"
    mkdir -p  $PARENT_DIR/$rounds
    echo "Train the agent... ))"
    python main.py play --agents baby_terminator --n-rounds=$CHECKPOINT --train 1 --scenario coin-heaven --no-gui
    sleep 30
    echo "Training finished"
    echo "copy the old model"
    cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$currentdatetime/$rounds/
    echo "evaluate the agent"
    python evaluations_$AGENT.py
    cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
    echo "---------------------------------------------------"
done




# currentdatetime=$(date +"%Y%m%d%H%M")
# for idx in {1..6}
# do
#     rounds=$(( CHECKPOINT * idx ))
#     echo "new upper bound is ${rounds} rounds"
#     mkdir -p  $PARENT_DIR/$currentdatetime/$rounds
#     echo "Train the agent... ))"
#     python main.py play --agents baby_terminator peaceful_agent peaceful_agent coin_collector_agent --n-rounds=$CHECKPOINT --train 1 --scenario coin-heaven --no-gui
#     sleep 30
#     echo "Training finished"
#     echo "copy the old model"
#     cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$currentdatetime/$rounds/
#     echo "evaluate the agent"
#     python evaluations_$AGENT.py
#     cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
#     echo "---------------------------------------------------"
# done

# currentdatetime=$(date +"%Y%m%d%H%M")
# for idx in {1..8}
# do
#     rounds=$(( CHECKPOINT * idx ))
#     echo "new upper bound is ${rounds} rounds"
#     mkdir -p  $PARENT_DIR/$currentdatetime/$rounds
#     echo "Train the agent... ))"
#     python main.py play --agents baby_terminator --n-rounds=$CHECKPOINT --train 1 --scenario loot-crate --no-gui
#     sleep 30
#     echo "Training finished"
#     echo "copy the old model"
#     cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$currentdatetime/$rounds/
#     echo "evaluate the agent"
#     python evaluations_$AGENT.py
#     cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
#     echo "---------------------------------------------------"
# done

# currentdatetime=$(date +"%Y%m%d%H%M")
# for idx in {1..20}
# do
#     rounds=$(( CHECKPOINT * idx ))
#     echo "new upper bound is ${rounds} rounds"
#     mkdir -p  $PARENT_DIR/$currentdatetime/$rounds
#     echo "Train the agent... ))"
#     python main.py play --agents baby_terminator peaceful_agent peaceful_agent coin_collector_agent --n-rounds=$CHECKPOINT --train 1 --scenario loot-crate --no-gui
#     sleep 30
#     echo "Training finished"
#     echo "copy the old model"
#     cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$currentdatetime/$rounds/
#     echo "evaluate the agent"
#     python evaluations_$AGENT.py
#     cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
#     echo "---------------------------------------------------"
# done

# currentdatetime=$(date +"%Y%m%d%H%M")
# for idx in {1..20}
# do
#     rounds=$(( CHECKPOINT * idx ))
#     echo "new upper bound is ${rounds} rounds"
#     mkdir -p  $PARENT_DIR/$currentdatetime/$rounds
#     echo "Train the agent... ))"
#     python main.py play --agents baby_terminator rule_based_agent rule_based_agent rule_based_agent --n-rounds=$CHECKPOINT --train 1 --scenario loot-crate --no-gui
#     sleep 30
#     echo "Training finished"
#     echo "copy the old model"
#     cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$currentdatetime/$rounds/
#     echo "evaluate the agent"
#     python evaluations_$AGENT.py
#     cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
#     echo "---------------------------------------------------"
# done

# currentdatetime=$(date +"%Y%m%d%H%M")
# for idx in {1..30}
# do
#     rounds=$(( CHECKPOINT * idx ))
#     echo "new upper bound is ${rounds} rounds"
#     mkdir -p  $PARENT_DIR/$currentdatetime/$rounds
#     echo "Train the agent... ))"
#     python main.py play --agents baby_terminator rule_based_agent rule_based_agent rule_based_agent --n-rounds=$CHECKPOINT --train 1 --scenario classic --no-gui
#     sleep 30
#     echo "Training finished"
#     echo "copy the old model"
#     cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$currentdatetime/$rounds/
#     echo "evaluate the agent"
#     python evaluations_$AGENT.py
#     cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
#     echo "---------------------------------------------------"
# done


# for idx in {1..5}
# do
#     rounds=$(( CHECKPOINT * idx ))
#     echo "new upper bound is ${rounds} rounds"
#     mkdir -p  $PARENT_DIR/$rounds
#     echo "Train the agent... ))"
#     python main.py play --agents baby_terminator peaceful_agent peaceful_agent coin_collector_agent --n-rounds=$CHECKPOINT --train 1 --scenario loot-crate --no-gui
#     pid=$(pgrep -f "python main.py play")
#     tail --pid="$pid" -f /dev/null
#     echo "Training finished"
#     echo "copy the old model"
#     cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$rounds/
#     echo "evaluate the agent"
#     python evaluations_$AGENT.py
#     cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
#     echo "---------------------------------------------------"
# done



# for idx in {1..5}
# do
#     rounds=$(( CHECKPOINT * idx ))
#     echo "new upper bound is ${rounds} rounds"
#     mkdir -p  $PARENT_DIR/$rounds
#     echo "Train the agent... ))"
#     python main.py play --agents baby_terminator coin_collector_agent coin_collector_agent coin_collector_agent --n-rounds=$CHECKPOINT --train 1 --scenario loot-crate --no-gui
#     pid=$(pgrep -f "python main.py play")
#     tail --pid="$pid" -f /dev/null
#     echo "Training finished"
#     echo "copy the old model"
#     cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$rounds/
#     echo "evaluate the agent"
#     python evaluations_$AGENT.py
#     cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
#     echo "---------------------------------------------------"
# done



# for idx in {1..5}
# do
#     rounds=$(( CHECKPOINT * idx ))
#     echo "new upper bound is ${rounds} rounds"
#     mkdir -p  $PARENT_DIR/$rounds
#     echo "Train the agent... ))"
#     python main.py play --agents baby_terminator rule_based_agent coin_collector_agent coin_collector_agent --n-rounds=$CHECKPOINT --train 1 --scenario loot-crate --no-gui
#     pid=$(pgrep -f "python main.py play")
#     tail --pid="$pid" -f /dev/null
#     echo "Training finished"
#     echo "copy the old model"
#     cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$rounds/
#     echo "evaluate the agent"
#     python evaluations_$AGENT.py
#     cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
#     echo "---------------------------------------------------"
# done


# for idx in {1..5}
# do
#     rounds=$(( CHECKPOINT * idx ))
#     echo "new upper bound is ${rounds} rounds"
#     mkdir -p  $PARENT_DIR/$rounds
#     echo "Train the agent... ))"
#     python main.py play --agents baby_terminator rule_based_agent rule_based_agent rule_based_agent --n-rounds=$CHECKPOINT --train 1 --scenario loot-crate --no-gui
#     pid=$(pgrep -f "python main.py play")
#     tail --pid="$pid" -f /dev/null
#     echo "Training finished"
#     echo "copy the old model"
#     cp $PARENT_DIR/../../my-saved-model.pkl.gz $PARENT_DIR/$rounds/
#     echo "evaluate the agent"
#     python evaluations_$AGENT.py
#     cat agent_code/$AGENT/logs/$AGENT.log >> agent_code/$AGENT/logs/all.log
#     echo "---------------------------------------------------"
# done
