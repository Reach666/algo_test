import random
import gymnasium as gym
import numpy as np
from collections import deque
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 队列,先进先出

    def add(self, obs, action, reward, next_obs, done):  # 将数据加入buffer
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*transitions)
        return np.array(obs), action, reward, np.array(next_obs), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 隐藏层使用ReLU激活函数
        return self.fc3(x)

class VAnet(nn.Module):
    ''' Dueling DQN = V + A '''
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)  # 共享网络部分
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_A = nn.Linear(hidden_dim, action_dim)
        self.fc_V = nn.Linear(hidden_dim, 1)

        # self.fc_A = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_dim, action_dim),
        # )
        # self.fc_V = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_dim, 1),
        # )

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 隐藏层使用ReLU激活函数
        A = self.fc_A(x)
        V = self.fc_V(x)
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到，A的均值为0
        return Q

class DQN:
    ''' DQN算法 '''
    def __init__(self, obs_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device, dqn_type='DuelingDQN'):
        self.dqn_type = dqn_type
        self.action_dim = action_dim
        if self.dqn_type == 'DuelingDQN': # Dueling DQN采取不一样的网络框架
            self.q_net = VAnet(obs_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = VAnet(obs_dim, hidden_dim, self.action_dim).to(device)
        else:
            self.q_net = Qnet(obs_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = Qnet(obs_dim, hidden_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate, weight_decay=0)  # 使用Adam优化器  # weight_decay=1e-3
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, obs):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            obs = torch.tensor(np.array([obs]), dtype=torch.float).to(self.device)
            action = self.q_net(obs).argmax().item()
        return action

    def max_q_value(self, obs):
        obs = torch.tensor(np.array([obs]), dtype=torch.float).to(self.device)
        return self.q_net(obs).max().item()

    def update(self, transition_dict):
        obs = torch.tensor(transition_dict['obs'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_obs = torch.tensor(transition_dict['next_obs'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(obs).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q_net(next_obs).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_obs).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_obs).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1


# 测试代理
def test(agent,steps=None):
    agent_epsilon = agent.epsilon
    agent.epsilon = 0
    env_t = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env_t.reset()
    _steps = steps if steps is not None else 1000000
    for i in range(_steps):
        action = agent.take_action(obs)
        obs, reward, terminated, truncated, _ = env_t.step(action)
        if terminated or truncated:
            if steps is None:
                break
            obs, _ = env_t.reset()
        # env_t.render()  # time.sleep(0.01)
    env_t.close()
    agent.epsilon = agent_epsilon


def train_DQN(agent, env, num_episodes, replay_buffer, q_update, minimal_size, batch_size):
    scheduler = ReduceLROnPlateau(agent.optimizer, mode='max', factor=0.5, patience=100, verbose=True)  # 其实强化学习的学习率不太适合衰减，因为随时都可能找到新策略打开新世界
    filtered_episode_return = None
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    i_step = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                obs, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(obs)
                    max_q_value = agent.max_q_value(obs) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(obs, action, reward, next_obs, done)
                    obs = next_obs
                    episode_return += reward
                    i_step += 1
                    if i_step % q_update == 0:
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {
                                'obs': b_s,
                                'actions': b_a,
                                'next_obs': b_ns,
                                'rewards': b_r,
                                'dones': b_d
                            }
                            agent.update(transition_dict)
                return_list.append(episode_return)
                filtered_episode_return = 0.9 * filtered_episode_return + 0.1 * episode_return if filtered_episode_return is not None else episode_return
                scheduler.step(filtered_episode_return)

                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
                pbar.update(1)
                if (i_episode + 1) % 20 == 0:
                    test(agent)
    return return_list, max_q_value_list





lr = 2e-3
num_episodes = 500
hidden_dim = 32
gamma = 0.99
epsilon = 0.2
q_update = 3  # 每q_update个env的step，q参数训练一次
target_update = 10  # 每target_update次q训练，target更新一次
buffer_size = 4096 * 100  # 一般越大越好
batch_size = 256  # 从buffer中每次抽取的数量，感觉可以更大
minimal_size = max(batch_size,512)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('device:',device)

env_name = 'CartPole-v1'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(obs_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, dqn_type='DuelingDQN')  # DQN  DoubleDQN  DuelingDQN
return_list, max_q_value_list = train_DQN(agent, env, num_episodes, replay_buffer, q_update, minimal_size, batch_size)



episodes_list = list(range(len(return_list)))
# mv_return = rl_utils.moving_average(return_list, 5)  # import rl_utils
plt.figure()
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')  # 500是满分
plt.title('DQN on {}'.format(env_name))

frames_list = list(range(len(max_q_value_list)))
plt.figure()
plt.plot(frames_list, max_q_value_list)
# plt.axhline(0, c='orange', ls='--')
# plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('DQN on {}'.format(env_name))
plt.show()


test(agent,steps=1000)



