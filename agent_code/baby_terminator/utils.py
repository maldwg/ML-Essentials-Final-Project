import torch
from collections import namedtuple, deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
device = torch.device("cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))