# %%  Import
import os
import copy
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

USE_GENERATED_MEMORY = False

#%% hyper parameters
EPS_START = 0.95  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 50000  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 256  # NN hidden layer size
BATCH_SIZE = 350  # Q-learning batch size


#establish the environment
#env = gym.make('CartPole-v0') 
env = gym.make('2048-v0')

#%% DQN NETWORK ARCHITECTURE
#model = DQN(4, 4, 4)
model = torch.load('pretrained_model/current_model_30000.pth') #DQN(4, 4, 4)#DuelingDQN(env.observation_space.shape, 4)
model.cuda()
target = torch.load('pretrained_model/current_model_30000.pth')#DQN(4, 4, 4)#DuelingDQN(env.observation_space.shape, 4)
target.cuda()
target.eval()
target.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), LR)

#%% SELECT ACTION USING GREEDY ALGORITHM
steps_done = 0
def select_action(state):
    global steps_done
    global no_update_for_episode
    #global epsHistory 
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if no_update_for_episode:
        epsHistory.append(eps_threshold)

    #print(state.shape)
    #print(eps_threshold)
    if sample > eps_threshold:
        #return argmaxQ
        with torch.no_grad():
            return model(state).max(1)[1].view(1, 1).cpu()
    else:
        #return random action
        return torch.LongTensor([[random.randrange(2)]])
    


#%% Setup Memory size
if USE_GENERATED_MEMORY:
    print("using memory")
    memory = pickle.load(open("replay_memory.p", 'rb'))
else:
    memory = ReplayMemory(50000000)

def run_episode(e, environment):
    global total_rewards
    global episode_durations
    global action_history
    global no_update_for_episode

    state = environment.reset()
    #state = state.flatten()
    steps = 0
    total_reward = 0
    while True:
        #environment.render()
        # state = state/131072 #.flatten()
        action = select_action(torch.FloatTensor([state]))
        next_state, reward, done, _ = environment.step(action.numpy()[0, 0])
        action_history[action.numpy()[0, 0]] += 1
        total_reward = total_reward + reward
        if done:
            #print(next_state)
            print("Reward: {0} || Episode {1} finished after {2} steps".format(total_reward, e, steps))
            reward = -1
            total_reward = total_reward + reward
            total_rewards.append(total_reward)
            total_reward = 0
        if (state == next_state).all():
            reward = -100
        memory.push( state,
                     action.numpy()[0,0],
                     next_state,
                     reward)
        learn()
        state = next_state

        steps += 1

        if done:
            #print("{2} Episode {0} finished after {1} steps".format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            
            episode_durations.append(steps)
            

            break
        no_update_for_episode = True
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
    current_q_values = model(batch_state)
    current_q_values = current_q_values.gather(1, batch_action.cuda())
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = target(batch_next_state).detach().max(1)[0]
    expected_q_values = (batch_reward + (GAMMA * max_next_q_values.cpu())).unsqueeze(1)

    # loss is measured from error between current and newly expected Q values
    loss = F.mse_loss(current_q_values.cuda(), expected_q_values.cuda())

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if loss < 100:
        losses.append(loss.data)
#%% RUN AND SHOW THE RESULT

EPISODES = 15000  # number of episodes



if not os.path.exists('pretrained_model/'):
    os.mkdir('pretrained_model/')


#%% Run episodes learning
episode_durations = []
total_rewards = []
epsHistory = []
losses = []
action_history=[0]*4
ACTION_STRING = ['left','up','right','down']
no_update_for_episode = False
for e in range(EPISODES):
    no_update_for_episode = False
    run_episode(e, env)
    if e % 10 == 0:
        target.load_state_dict(model.state_dict())
    if e % 5000 == 0 and e != 0:
        torch.save(target, "pretrained_model/current_model_" + str(e + 30000) + ".pth")

        fig, ax = plt.subplots()
        x = np.array(range(e + 1))
        y = np.array(total_rewards)
        ax.plot(x, y)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Total rewards')
        fig.savefig('images/Rewards_{0}.jpg'.format(e + 30000))
        plt.close(fig)

        fig, ax = plt.subplots()
        y = np.array(epsHistory)
        ax.plot(range(len(epsHistory)), y)
        ax.set_xlabel('Number iterations')
        ax.set_ylabel('Eps history')
        fig.savefig('images/Eps_History_{0}.jpg'.format(e + 30000))
        plt.close(fig)

        fig, ax = plt.subplots()
        y = np.array(episode_durations)
        ax.plot(x, y)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Number of steps')
        fig.savefig('images/Steps_{0}.jpg'.format(e + 30000))
        plt.close(fig)

        fig, ax = plt.subplots()
        y = np.array(losses)
        ax.plot(range(len(losses)), y)
        ax.set_xlabel('Number iterations')
        ax.set_ylabel('LOSS')
        fig.savefig('images/Loss_{0}.jpg'.format(e + 30000))
        plt.close(fig)

        plt.bar(np.arange(len(ACTION_STRING)), action_history, align='center')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(len(ACTION_STRING)), ACTION_STRING)
        plt.title("Action history")
        plt.savefig('images/Actions_{0}.jpg'.format(e+ 30000))


print('Complete')
plt.plot(total_rewards)
plt.plot(epsHistory)
plt.plot(episode_durations)
plt.plot(losses)
plt.show()