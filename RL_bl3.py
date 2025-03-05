import gymnasium as gym
from gymnasium import RewardWrapper
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import time
import numpy as np


# 修改默认奖励策略
class PosBasedRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.x_initial_area = [float('-0.6'),float('-0.4')]
        self.x_area = list(self.x_initial_area)
        self.v_mem = np.zeros(18)

    def reset(self, **kwargs):  # 开局重置奖励规则
        self.x_area = list(self.x_initial_area)
        self.v_mem = np.zeros(18)
        return super().reset(**kwargs)

    def reward(self, reward):
        # 获取当前观测值，obs[0]是位置，obs[1]是速度   x取值范围[-1.2, 0.6]   v取值范围[-0.07, 0.07]   位置初始化在[-0.6, -0.4]
        x, v = self.env.state

        bonus = 0
        # 超速奖励  效果明显好于速度奖励
        ind = int(x*10+12)
        if abs(v) > self.v_mem[ind]:
            bonus += (v**2 - self.v_mem[ind]**2) * 10000
            self.v_mem[ind] = abs(v)
        # # 位置离目标越近奖励越多
        # bonus += -abs(x - 0.5) * 0.1
        # # 探索新区间奖励
        # if x < self.x_area[0]:
        #     bonus += (self.x_area[0] - x) * 50
        #     self.x_area[0] = x
        # if x > self.x_area[1]:
        #     bonus += (x - self.x_area[1]) * 100
        #     self.x_area[1] = x

        # print(x, v, reward, bonus,reward + bonus)
        return reward + bonus  # 叠加额外奖励
env = gym.make("MountainCar-v0", render_mode=None)
env = PosBasedRewardWrapper(env)
env = DummyVecEnv([lambda: env])


# 创建并包装环境
# env = DummyVecEnv([lambda: gym.make("CartPole-v1", render_mode=None)])
# env = DummyVecEnv([lambda: gym.make("MountainCar-v0", render_mode=None)])  # render_mode="rgb_array"


# 创建PPO代理
# model = DQN("MlpPolicy", env, batch_size=32, learning_rate=1e-4, exploration_final_eps=0.05,verbose=1)
# model = A2C("MlpPolicy", env, verbose=1)
model = PPO("MlpPolicy", env, n_steps=2048, gamma=0.99, ent_coef=0.0, vf_coef=0.5,verbose=1)
# 训练代理
model.learn(total_timesteps=200000)
env.close()

# 测试代理
# env_t = DummyVecEnv([lambda: gym.make("CartPole-v1", render_mode="human")])
env_t = DummyVecEnv([lambda: gym.make("MountainCar-v0", render_mode="human")])

obs = env_t.reset()  # 注意：这里不再返回info元组，而是只返回obs
for i in range(1000):
    action, _states = model.predict(obs)  # 直接传递obs
    obs, rewards, dones, info = env_t.step(action)
    # env_t.render()
    if dones:
        obs = env_t.reset()
    # time.sleep(0.01)
env_t.close()
