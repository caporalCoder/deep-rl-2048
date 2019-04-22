# %%  Import
import os
import gym_2048
import gym
import random
import math
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from network import Network, DQN, DuelingDQN
from memory import ReplayMemory
import numpy as np
#%% hyper parameters
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 256  # NN hidden layer size
BATCH_SIZE = 64  # Q-learning batch size


#establish the environment
#env = gym.make('CartPole-v0') 
env = gym.make('2048-v0')

#%% DQN NETWORK ARCHITECTURE
#model = DQN(4, 4, 4)
model = DuelingDQN(env.observation_space.shape, 4)
model.cuda()
optimizer = optim.Adam(model.parameters(), LR)

#%% SELECT ACTION USING GREEDY ALGORITHM
steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    #print(state.shape)
    #print(eps_threshold)
    if sample > eps_threshold:
        #return argmaxQ
        # state = state.cuda()
        # Variable(state, volatile=True).type(torch.FloatTensor)
        return model(state).data.max(1)[1].view(1, 1).cpu()
    else:
        #return random action
        return torch.LongTensor([[random.randrange(2)]])
    


#%% Setup Memory size
memory = ReplayMemory(10000)
episode_durations = []
def run_episode(e, environment):
    state = environment.reset()
    #state = state.flatten()
    steps = 0
    total_reward = 0
    while True:
        #environment.render()
        state = state/np.amax(state) #.flatten()
        action = select_action(torch.FloatTensor([state]))
        
        next_state, reward, done, _ = environment.step(action.numpy()[0, 0])
        # negative reward when attempt ends
        total_reward = total_reward + reward
        if done:
            #print(next_state)
            print("Reward: {0} || Episode {1} finished after {2} steps".format(total_reward, e, steps))
            total_reward = 0
            reward = -10
        next_state = next_state/np.amax(next_state) #.flatten()
        
        memory.push((torch.FloatTensor([state]),
                     action,  # action is already a tensor
                     torch.FloatTensor([next_state]),
                     torch.FloatTensor([reward])))

        learn()

        state = next_state
        steps += 1

        if done:
            #print("{2} Episode {0} finished after {1} steps".format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            
            episode_durations.append(steps)

            if steps % 100 == 0:
                torch.save(model, "pretrained_model/current_model_" + str(steps) + ".pth")

            break
            
#%% TRAIN THE MODEL
def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
    
    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(Variable(batch_state, volatile=True).type(torch.cuda.FloatTensor))
    current_q_values = current_q_values.gather(1, batch_action.cuda())
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values.cpu())

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values.cuda().reshape_as(expected_q_values.cuda()), expected_q_values.cuda())

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#%% RUN AND SHOW THE RESULT

EPISODES = 10000  # number of episodes



if not os.path.exists('pretrained_model/'):
    os.mkdir('pretrained_model/')


#%% Run episodes learning
for e in range(EPISODES):
    run_episode(e, env)

print('Complete')
plt.plot(episode_durations)
plt.show()