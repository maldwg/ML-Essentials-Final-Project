import json
import subprocess
import gzip
import pickle

from copy import deepcopy
from itertools import product

import numpy as np

import hyperparameters


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


def search():
    top_ten_mean_rewards = []
    for json_dump in create_dicts():
        with open("./parameters.json", "w") as f:
            f.write(json_dump)

        # call main.py
        p_main = subprocess.Popen("python main.py play --agents baby_terminator --n-rounds=200 --train 1 --scenario coin-heaven --no-gui")
        exit_code = p_main.wait()
        # eval
        if exit_code == 0:
            with gzip.open("./my-saved-model.pkl.gz", 'rb') as f:
                _, _, _, memory = pickle.load(f)
            # avg reward berechnen
            mean_reward = np.mean(memory.rewards_of_round[-10:])
            # wenn top ten, wegkopieren
            top_ten_mean_rewards.append(mean_reward)
            if len(top_ten_mean_rewards) == 10:
                top_ten_mean_rewards.pop(min(top_ten_mean_rewards))
                


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
