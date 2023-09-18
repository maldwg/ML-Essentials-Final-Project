import json

from enum import Enum


class Convolution(Enum):
    OUTPUT_CHANNELS = "output_channels"
    KERNEL_SIZE = "kernel_size"
    STRIDE = "stride"
    PADDING = "padding"
    DROPOUT = "dropout"


class Optimizer(Enum):
    LEARNING_RATE = "lr"
    WEIGHT_DECAY = "weight_decay"


class Memory(Enum):
    REPLAY_MEMORY_SIZE = "capacity"    


class EpsilonGreedy(Enum):
    EPS_START = "EPS_START"
    EPS_END = "EPS_END"
    EPS_DECAY = "EPS_DECAY"


class Train(Enum):
    BATCH_SIZE = "BATCH_SIZE"
    GAMMA = "GAMMA"


def read_hyperparameters():
    with open("./parameters.json", "r") as f:
       hyperparameters = json.load(f)
    return hyperparameters