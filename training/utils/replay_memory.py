import random
from collections import namedtuple, deque
# Initialize transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')) #, 'explore'
Transition.__new__.__defaults__ = (None,) * len(Transition._fields)
# #%% MEMORY REPLAY
# class ReplayMemory:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []

#     def push(self, transition):
#         self.memory.append(transition)
#         if len(self.memory) > self.capacity:
#             del self.memory[0]

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)
# class ReplayMemory(object):
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def push(self, *args):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)

from collections import deque, namedtuple
import random

# class Transition:
#     def __init__(self, *args):
#         assert len(args) == 4
#         self.state, self.action, self.next_state, self.reward = args
#     def __repr__(self):
#         return 'state: {}, action: {}, next_state: {}, reward : {}'.format(
#             self.state, self.action, self.next_state, self.reward)

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, max_length):
        self.max_length = max_length
        self.memory = deque()

    def push(self, *args):
        while len(self.memory) >= self.max_length:
            self.memory.popleft()
        entry = Transition(*args)
        self.memory.append(entry)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __repr__(self):
        if len(self) == 0:
            return 'ReplayMemory.memory: EMPTY'
        return 'ReplayMemory.memory: {}...'.format(self.memory[0])

    def __len__(self):
        return len(self.memory)

