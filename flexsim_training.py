import gym
from flexsim_env import FlexSimEnv
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3 import PPO
from stable_baselines3 import A2C
#from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
# Get our repository
!git clone https://nab170130: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@github.com/saodem74/Transfer-Learning-in-Reinforcement-Learning.git


def main():
    print("Initializing FlexSim environment...")

    # Create a FlexSim OpenAI Gym Environment
    env = FlexSimEnv(
        flexsimPath = "C:/Program Files/FlexSim 2022/program/flexsim.exe",
        #需要更改
        modelPath = "C:/Users/mark/Documents/FlexSim 2022 Projects/tutorials/FlexSim 2022/ChangeoverTimesRL.fsm",
        verbose = False,
        visible = False
        )
    check_env(env) # Check that an environment follows Gym API.


    # Training a baselines3 PPO model in the environment
    #model = PPO("MlpPolicy", env, verbose=1)
    model = A2C("MlpPolicy", env, verbose=1)
    #model = DQN("MlpPolicy", env, verbose=1)
    print("Training model...")
    model.learn(total_timesteps=1000000
                )
   
    
    
    # save the model
    print("Saving model...")
    #需要更改
    model.save("ChangeoverTimesRL")

    input("Waiting for input to do some test runs...")

    # Run test episodes using the trained model
    for i in range(2):
        env.seed(i)
        
        observation = env.reset()
        env.render()
        done = False
        rewards = []
        while not done:
            action, _states = model.predict(observation)
            observation, reward, done, info = env.step(action)
            env.render()
            rewards.append(reward)
            if done:
                cumulative_reward = sum(rewards)
                print("Reward: ", cumulative_reward, "\n")
    env._release_flexsim()
    input("Waiting for input to close FlexSim...")
    env.close()


if __name__ == "__main__":
    main()