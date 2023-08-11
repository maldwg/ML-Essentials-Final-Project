AGENT=baby_terminator
N_ROUNDS=100
while getopts a:n: flag
do 
    case "${flag}" in
        a) AGENT=${OPTARG};;
        n) N_ROUNDS=${OPTARG};;
    esac
done
TRAIN_FILE=./agent_code/"$AGENT"/my-saved-model.pkl.gz
if test -f "$TRAIN_FILE"; then
    rm "$TRAIN_FILE"
fi
python main.py play --agents baby_terminator rule_based_agent rule_based_agent rule_based_agent --n-rounds="$N_ROUNDS" --train 1  --no-gui &&
python evaluations.py
