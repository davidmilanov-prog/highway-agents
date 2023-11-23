import gymnasium as gym
import highway_env
import stable_baselines3
# from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import tensorboard 
import os
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

MODEL_TEST_PATH = "snn_model/snn_model.pth"
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.3 # Decay rate
threshold = 0.5 # When SNN Spikes

def create_model(beta, threshold):
    net = nn.Sequential(
        nn.Conv2d(1, 8, 3),  # 10x10
        nn.Upsample(scale_factor=2, mode='nearest'),  # 20x20
        snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True),
        
        nn.Conv2d(8, 16, 5),  # 16x16
        snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True),
        
        nn.Flatten(),
        nn.Linear(16 * 16 * 16, 1),
        snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True, output=True)
    ).to(device)
    return net

def test():
    env = gym.make("racetrack-v0", render_mode="rgb_array")
    net = create_model(beta, threshold)

    net.load_state_dict(torch.load(MODEL_TEST_PATH, map_location=torch.device('cpu')))

    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for timestep in range(1000):
            done = truncated = False
            obs, info = env.reset() 
            while not (done or truncated):
                data = torch.tensor(obs[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add batch dimension and send to device

                _, output_tuple = net(data)
                output = output_tuple[0]
                output_numpy = output.cpu().numpy()
                obs, reward, done, truncated, info = env.step(output_numpy)
                if obs[1].sum() == 0:
                    done = True

                # Render
                env.render()
    env.close()

if __name__ == "__main__":  
    test()