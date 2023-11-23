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

TRAIN = False
# Function to automatically generate model names
def generate_model_names(base_path, train, choice=None):
    existing_models = [file for file in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, file))]
    if existing_models:
        latest_model = max(existing_models)
        latest_version = int(latest_model.split("model")[1]) + 1
    else:
        latest_version = 1

    if train:
        model_name = f"model{latest_version}"
    else:
        if choice is not None:
            model_name = "model"+str(choice)
        else:
            model_name = latest_model if existing_models else "model"
        
    model_path = os.path.join(base_path, model_name)
    return model_path

def train():
    
    n_cpu = 7
    batch_size = 64
    env = make_vec_env("racetrack-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env("racetrack-v0")
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.9,
                verbose=2,
                tensorboard_log=MODEL_TRAIN_PATH
                )
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=os.path.join(MODEL_TRAIN_PATH,"best_model"), 
                                 log_path=MODEL_TRAIN_PATH, 
                                 eval_freq=5000, 
                                 n_eval_episodes=5, 
                                 deterministic=True, 
                                 render=False
                                 )
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e5), callback=eval_callback)
        model.save(MODEL_NAME_TRAIN_PATH)
        del model

def test():
  env = gym.make("racetrack-v0", render_mode="rgb_array")
  model=PPO.load(MODEL_NAME_TEST_PATH)
  for timestep in range(1000):
      done = truncated = False
      obs, info = env.reset() 
      while not (done or truncated):
          # Predict
          action, _states = model.predict(obs, deterministic=True)
          # Get reward
          obs, reward, done, truncated, info = env.step(action)
          print(obs)
          print(obs.shape)
          # Render
          env.render()
  env.close()
if __name__ == "__main__":
    if TRAIN:
        MODEL_TRAIN_PATH = generate_model_names("racetrack_ppo", train=TRAIN, choice=None)
        MODEL_NAME_TRAIN_PATH = os.path.join(MODEL_TRAIN_PATH, "model")
        train()
    else:
        MODEL_TEST_PATH = generate_model_names("racetrack_ppo", train=TRAIN, choice=1)
        MODEL_NAME_TEST_PATH = os.path.join(MODEL_TEST_PATH, "model")
        test()