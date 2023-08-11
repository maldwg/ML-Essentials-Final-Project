DEFAULTAGENT=baby_terminator
AGENT="${1:$DEFAULTAGENT}"
DEFAULTROUNDS=100
N_ROUNDS="${2:-$DEFAULTROUNDS}"

TRAIN_FILE=./agent_code/"$DEFAULTAGENT"/my-saved-model.pkl.gz
if test -f "$TRAIN_FILE"; then
    rm "$TRAIN_FILE"
fi
python main.py play --agents baby_terminator rule_based_agent rule_based_agent rule_based_agent --n-rounds="$N_ROUNDS" --train 1  --no-gui &&
python evaluations.py
