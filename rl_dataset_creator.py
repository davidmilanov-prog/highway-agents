import gymnasium as gym
import highway_env
import stable_baselines3
from stable_baselines3 import PPO
import os
import h5py
import numpy as np

# Remember to change number when trying to access different models
MODEL_TEST_PATH = "racetrack_ppo/model1"
MODEL_NAME_TEST_PATH = os.path.join(MODEL_TEST_PATH, "model")
DATASET_NAME = "datasets/long_ppo_data"

def create_dataset():
  env = gym.make("racetrack-v0", render_mode="rgb_array")
  model=PPO.load(MODEL_NAME_TEST_PATH)
  # Data variables

  car_observations = []
  lane_observations = []
  intermediate_car_observations = []
  intermediate_lane_observations = []
  all_actions = []
  intermediate_actions = []
  for episode in range(100):
      print(f'Episode: {episode}')
      done = truncated = False
      obs, info = env.reset() 
      timestep = 0
      while not (done or truncated):
          # Predict
          action, _states = model.predict(obs, deterministic=True)
          # Get reward
          obs, reward, done, truncated, info = env.step(action)
          # Render
          env.render()
          # Collect data
          intermediate_car_observations.append(obs[0])
          intermediate_lane_observations.append(obs[1])
          intermediate_actions.append(action)
          timestep+=1
      if(timestep>=299):
         car_observations.extend(intermediate_car_observations)
         lane_observations.extend(intermediate_lane_observations)
         all_actions.extend(intermediate_actions)
      if len(all_actions) > 100000:
         break
  
  env.close()
  # We are combining both obs arrays, need to make sure we have TWO 12x12 lists.
  # Make sure they are in the exact same structure
  car_observations = np.stack(car_observations, axis=0)
  lane_observations = np.stack(lane_observations, axis=0)
  all_actions = np.stack(all_actions, axis=0)
  with h5py.File(DATASET_NAME, 'w') as hf:
      hf.create_dataset("car tracker", data=car_observations)
      hf.create_dataset("lane tracker", data=lane_observations)
      hf.create_dataset("actions", data=all_actions)
   

if __name__ == "__main__":
  create_dataset()