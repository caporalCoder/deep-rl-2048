from torch import nn
import torch.nn.functional as F
import torch as T
import numpy as np
class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.input_layer = nn.Linear(16, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(F.sigmoid(x))
        return x

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=2, stride=2)
        #self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        #self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=2, stride=2)
        #self.bn3 = nn.BatchNorm2d(32)

        #self.loss = nn.MSELoss()
        if T.cuda.is_available():
            self.device = T.cuda.set_device(0)
        else:
            self.device = T.device('cpu')
        self.to(self.device)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=2, stride=2):
            return ((size - (kernel_size - 1) - 1) // stride) + 1

        convw = 2 #conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = 2 #conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.linear_input_size = 256#convw * convh * 512
        self.fc1 = nn.Linear(self.linear_input_size, 256)
        self.head = nn.Linear(256, outputs)

    def forward(self, observation):
        observation = observation.cuda()#T.Tensor(observation).to(self.device)
        observation = observation.view(-1, 1, 4, 4)
        observation = F.relu(self.conv1(observation))

        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.fc1(observation.view(-1, self.linear_input_size)))
        return self.head(observation)


class DuelingDQN(nn.Module):
    def __init__(self, shape, outputs):
        super(DuelingDQN, self).__init__()
        
    
        self.input_shape = shape
        self.num_actions = outputs
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1)

        if T.cuda.is_available():
            self.device = T.cuda.set_device(0)
        else:
            self.device = T.device('cpu')
        self.to(self.device)

        self.adv1 = nn.Linear(64, 512)
        self.adv2 = nn.Linear(512, self.num_actions)

        self.val1 = nn.Linear(64 , 512)
        self.val2 = nn.Linear(512, 1)
        
    def forward(self, x):
        x = x.cuda()
        x = x.view(-1, 1 , 4, 4)
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = x.view(-1, 64)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()
    
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(T.zeros(1, *self.input_shape)))).view(1, -1).size(1)


