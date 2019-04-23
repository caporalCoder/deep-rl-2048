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
from utils.replay_memory import ReplayMemory, Transition
import numpy as np

USE_GENERATED_MEMORY = True

#%% hyper parameters
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 256  # NN hidden layer size
BATCH_SIZE = 16  # Q-learning batch size


#establish the environment
#env = gym.make('CartPole-v0') 
env = gym.make('game-2048-v0')

#%% DQN NETWORK ARCHITECTURE
#model = DQN(4, 4, 4)
model = DuelingDQN(env.observation_space.shape, 4)
model.cuda()
optimizer = optim.Adam(model.parameters(), LR)

#%% SELECT ACTION USING GREEDY ALGORITHM
steps_done = 0
def select_action(state):
    global steps_done
    #global epsHistory 
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    #epsHistory.append(eps_threshold)

    #print(state.shape)
    #print(eps_threshold)
    # if sample > eps_threshold:
        #return argmaxQ
        # state = state.cuda()
        # Variable(state, volatile=True).type(torch.FloatTensor)
    return model(Variable(state, volatile=True).type(torch.FloatTensor)).data.max(1)[1].view(1, 1).cpu()
    # else:
    #     #return random action
    #     return torch.LongTensor([[random.randrange(2)]])
    


#%% Setup Memory size
if USE_GENERATED_MEMORY:
    print("using memory")
    memory = pickle.load(open("replay_memory.p", 'rb'))
else:
    memory = ReplayMemory(50000000)
episode_durations = []
total_rewards = []
epsHistory = []
losses = []
def run_episode(e, environment):
    global total_rewards
    global episode_durations

    state = environment.reset()
    #state = state.flatten()
    steps = 0
    total_reward = 0
    while True:
        #environment.render()
        # state = state/131072 #.flatten()
        action = select_action(torch.FloatTensor([state]))
        
        next_state, reward, done, _ = environment.step(action.numpy()[0, 0])
        total_reward = total_reward + reward
        if done:
            print(next_state)
            print("Reward: {0} || Episode {1} finished after {2} steps".format(total_reward, e, steps))
            total_reward = 0
            reward = -10

        learn()

        state = next_state
        steps += 1

        if done:
            #print("{2} Episode {0} finished after {1} steps".format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            
            episode_durations.append(steps)
            total_rewards.append(reward)
            if steps % 100 == 0:
                torch.save(model, "pretrained_model/current_model_" + str(steps) + ".pth")

            break
            
#%% TRAIN THE MODEL
def learn():
    global losses
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    batch_state = Variable(torch.FloatTensor(batch.state))
    batch_action = Variable(torch.LongTensor(batch.action)).unsqueeze(1)
    batch_reward = Variable(torch.FloatTensor(batch.reward))
    batch_next_state = Variable(torch.FloatTensor(batch.next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(Variable(batch_state, volatile=True).type(torch.cuda.FloatTensor))
    current_q_values = current_q_values.gather(1, batch_action.cuda())
    # expected Q values are estimated from actions which gives maximum Q value
    max_ = model(batch_next_state).detach().max(1)
    #print(max_)
    max_next_q_values =  max_[0]
    #print(max_next_q_values)
    expected_q_values = batch_reward + (GAMMA * max_next_q_values.cpu())

    # loss is measured from error between current and newly expected Q values
    loss = F.mse_loss(current_q_values.cuda().reshape_as(expected_q_values.cuda()), expected_q_values.cuda())

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.data)
#%% RUN AND SHOW THE RESULT

EPISODES = 1000  # number of episodes



if not os.path.exists('pretrained_model/'):
    os.mkdir('pretrained_model/')


#%% Run episodes learning
for e in range(EPISODES):
    run_episode(e, env)

print('Complete')
plt.plot(total_rewards)
plt.plot(epsHistory)
plt.plot(episode_durations)
plt.plot(losses)
plt.show()