Help() 
{
    echo
    echo "Retrain a chosen agent and remove the original model weights if necessary."
    echo "Evaluation plots are provided inside the agent's folder."
    echo
    echo "Syntax: ./retrain.sh [-h|-a <agent_name>|-n <number_of_training_rounds>|-s <scenario_name>]"
    echo "Options:"
    echo "  -h                 Print this help"
    echo "  -c                 Continue training - Do not remove the old weights of the model."
    echo "  -a agent_name      Train the agent called 'agent_name'"
    echo "                                default: baby_terminator"
    echo "  -n 25              Train the agent in 25 rounds"
    echo "                                default: 100"
    echo "  -s coin-heaven     Train on the 'coin-heaven' scenario. Possible inputs are: empty, coin-heaven, loot-crate, classic."
    echo "                                default: classic"
}

AGENT=baby_terminator
N_ROUNDS=100
SCENARIO=coin-heaven
CONTINUE=false

while getopts a:n:s:ch flag; do 
    case "${flag}" in
        a) AGENT=${OPTARG};;
        n) N_ROUNDS=${OPTARG};;
        s) SCENARIO=${OPTARG};;
        c) CONTINUE=true;;
        h) Help && exit;;
    esac
done

TRAIN_FILE=./agent_code/"$AGENT"/my-saved-model.pkl.gz
if $(test -f "$TRAIN_FILE") && [ "$CONTINUE" == false ] ; then
    rm "$TRAIN_FILE"
    echo "trained weights removed"
fi
python main.py play --agents "$AGENT" rule_based_agent rule_based_agent rule_based_agent --n-rounds="$N_ROUNDS" --train 1  --no-gui --scenario "$SCENARIO" &&
python evaluations.py