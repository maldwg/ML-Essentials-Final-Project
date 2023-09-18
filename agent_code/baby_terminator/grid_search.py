import json

from copy import deepcopy
from itertools import product

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
    for json_dump in create_dicts():
        with open("./parameters.json", "w") as f:
            f.write(json_dump)

        # main.py aufrufen
        # eval
        # avg reward berechnen
        # wenn top ten, wegkopieren


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
