import gym_2048
import gym
import torch
import time
from network import DuelingDQN

ACTION_STRING = {
    0:'left',
    1:'up',
    2:'right',
    3:'down'
}
if __name__ == "__main__":
    env = gym.make('2048-v0')
    #env.seed(42)

    state = env.reset()
    env.render()

    done = False

    moves = 0
    model = torch.load('pretrained_model/current_model_40000.pth')#DuelingDQN(env.observation_space.shape, 4)
    model.cuda()
    #model.load_state_dict(torch.load('pretrained_model/current_model_700.pth'))
    model.eval()
    while not done:
        with torch.no_grad():
            val = model(torch.FloatTensor([state]))
            action =  val.max(1)[1].view(1, 1).cpu().numpy()[0,0] #env.np_random.choice(range(4), 1).item()
        next_state, reward, done, info = env.step(action)
        state = next_state
        moves +=1
        print(f"Next Action: {ACTION_STRING[action]} \n\n Reward:{reward}")
        env.render()
        time.sleep(1)
        print(f"\nTotal Moves: {moves}")