import json
import subprocess
import gzip
import pickle

import os
import shutil

from copy import deepcopy
from itertools import product

import numpy as np

import hyperparameters

BASE_PATH = "./agent_code/baby_terminator"


intervals = {
    hyperparameters.Convolution.OUTPUT_CHANNELS.value: [16],
    hyperparameters.Convolution.KERNEL_SIZE.value: [3],
    hyperparameters.Convolution.STRIDE.value: [1],
    hyperparameters.Convolution.PADDING.value: [2],
    hyperparameters.Convolution.DROPOUT.value: [0],

    hyperparameters.Optimizer.LEARNING_RATE.value: [1e-3, 1e-4, 1e-5, 1e-6],
    hyperparameters.Optimizer.WEIGHT_DECAY.value: [1e-4, 1e-5, 1e-6],

    hyperparameters.Memory.REPLAY_MEMORY_SIZE.value: [1e4, 1e5],

    hyperparameters.EpsilonGreedy.EPS_START.value: [0.5, 0.25, 0.1, 0.05],
    hyperparameters.EpsilonGreedy.EPS_END.value: [0.05, 0.01, 0.005],
    hyperparameters.EpsilonGreedy.EPS_DECAY.value: [100, 500, 1e3, 1e4],

    hyperparameters.Train.BATCH_SIZE.value: [128],
    hyperparameters.Train.GAMMA.value: [0.66, 0.9, 0.95, 0.999]
}

def delete_worst_model(model_id):
    # delete the now 11th placed model
    shutil.rmtree(f"{BASE_PATH}/top-ten-models/{model_id}")

def save_model(model_id, score):
    files_to_move = ["my-saved-model.pkl.gz", "parameters.json"]
    model_dir = f"{BASE_PATH}/top-ten-models/{model_id}"

    # create unique directory for model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # move saved model and hyperparameter in new positions
    for file in files_to_move:
        os.replace(f"{BASE_PATH}/{file}", f"{model_dir}/{file}")

    # write score to file
    with open(f"{model_dir}/score.txt", "w") as f:
        f.write(f"{score}")
        f.close()


def search():
    grid_search_model_id = 1
    print(f"Now testing model {grid_search_model_id}")
    top_ten_models = []
    for json_dump in create_dicts():
        with open(f"{BASE_PATH}/parameters.json", "w") as f:
            f.write(json_dump)
            f.close()
        # call main.py
        p_main = subprocess.Popen("python main.py play --agents baby_terminator --n-rounds=200 --train 1 --scenario coin-heaven --no-gui", shell=True)
        exit_code = p_main.wait()
        # eval
        if exit_code == 0:
            with gzip.open(f"{BASE_PATH}/my-saved-model.pkl.gz", 'rb') as f:
                _, _, _, memory = pickle.load(f)
            # avg reward berechnen
            mean_reward = np.mean(memory.rewards_after_round[-10:])
            # wenn top ten, wegkopieren

            # get worst model depending on the avg reward
            if len(top_ten_models) == 10:
                worst_model_id, worst_model_score = min(top_ten_models, key = lambda x: x[1])
                if mean_reward > worst_model_score:
                    print(f"model {grid_search_model_id} with score {mean_reward} is better than worst model {worst_model_id} with score {worst_model_score}, replacing... ")
                    # delete worst model and save new one
                    delete_worst_model(worst_model_id)
                    top_ten_models.remove((worst_model_id, worst_model_score))
                    top_ten_models.append((grid_search_model_id, mean_reward))
                    save_model(grid_search_model_id, mean_reward)
            else: 
                # if list not yet full, save the model
                top_ten_models.append((grid_search_model_id, mean_reward))
                save_model(grid_search_model_id, mean_reward)

            
        # increment model counter for next run
        grid_search_model_id += 1


def create_dicts():
    keys, values = zip(*intervals.items())
    for bundle in product(*values):
        d = dict(zip(keys, bundle))

        conv1_kwargs = filter_dict(d, hyperparameters.Convolution)   
        conv2_kwargs = deepcopy(conv1_kwargs)
        conv2_kwargs[hyperparameters.Convolution.OUTPUT_CHANNELS.value] *= 2
        optimizer_kwargs = filter_dict(d, hyperparameters.Optimizer)
        memory_kwargs = filter_dict(d, hyperparameters.Memory)
        epsilon_greedy_kwargs = filter_dict(d, hyperparameters.EpsilonGreedy)
        train_kwargs = filter_dict(d, hyperparameters.Train)

        yield json.dumps([conv1_kwargs, conv2_kwargs, optimizer_kwargs, memory_kwargs, epsilon_greedy_kwargs, train_kwargs])
        

def filter_dict(d, enum):
    return {member.value: d[member.value] for member in enum}
    

if __name__ == "__main__":
    search()
