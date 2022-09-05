import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


LOAD_MODEL = 0
SAVE_MODEL = 1
BATCH_SIZE = 512 # 大概需要2G的显存
EPOCHS = 10000 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 下载训练集
train_dataset = datasets.MNIST(root="./data/", train=True, transform=transforms.ToTensor(), download=False) # 图片里只有0和1 因此生成器输出层用sigmoid效果好
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


input_dim = 100
# G_sampler = lambda n: torch.Tensor(np.random.random((n, input_dim))*1)
G_sampler = lambda n: torch.Tensor(np.random.randn(n, input_dim)*1)

# ##### MODELS: Generator model and discriminator model

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 32 * 32)
#         self.br1 = nn.Sequential(
#             nn.BatchNorm1d(1024),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
#         self.br2 = nn.Sequential(
#             nn.BatchNorm1d(128 * 7 * 7),
#             nn.ReLU()
#         )
#         self.conv1 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
#             nn.Sigmoid()
#             # nn.Tanh()
#         )
#
#     def forward(self, x):
#         x = self.br1(self.fc1(x))
#         x = self.br2(self.fc2(x))
#         x = x.reshape(-1, 128, 7, 7)
#         x = self.conv1(x)
#         output = self.conv2(x)
#         return output
#
#
# # =================================================判别器================================================================
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 32, 5, stride=1),
#             nn.LeakyReLU(0.2)
#         )
#         self.pl1 = nn.MaxPool2d(2, stride=2)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, 5, stride=1),
#             nn.LeakyReLU(0.2)
#         )
#         self.pl2 = nn.MaxPool2d(2, stride=2)
#         self.fc1 = nn.Sequential(
#             nn.Linear(64 * 4 * 4, 1024),
#             nn.LeakyReLU(0.2)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(1024, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pl1(x)
#         x = self.conv2(x)
#         x = self.pl2(x)
#         x = x.view(x.shape[0], -1)
#         x = self.fc1(x)
#         output = self.fc2(x)
#         return output


class Generator(nn.Module): # 生成器比鉴别器更难训练，应提高网络复杂度和训练力度，势均力敌最好
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 120 * 5 * 5)
        self.conv1 = nn.ConvTranspose2d(120, 30, kernel_size=(3,3))
        self.conv2 = nn.ConvTranspose2d(30, 1, kernel_size=(5,5))
        self.bn1 = nn.BatchNorm1d(50) # BN对于生成器很重要
        self.bn2 = nn.BatchNorm1d(3000)
        self.convbn1 = nn.BatchNorm2d(30)
        # self.f = F.tanh
        self.f = F.leaky_relu

    def forward(self, x):
        in_size = x.size(0)
        out = self.fc1(x) # 1 * 20
        out = self.bn1(out)
        out = self.f(out)
        out = self.fc2(out)  # 1 * 300
        out = self.bn2(out)
        out = self.f(out)
        out = out.view(in_size, 120,5,5)  # 1 * 12 * 5 * 5
        out = F.upsample(out, scale_factor=2,mode='nearest') # 1 * 12 * 10 * 10
        out = self.conv1(out) # 1 * 10 * 12 * 12
        out = self.convbn1(out)
        out = self.f(out)
        out = F.upsample(out, scale_factor=2, mode='nearest')  # 1 * 10 * 24 * 24
        out = self.conv2(out)  # 1 * 28 * 28
        out = F.sigmoid(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 1*1*28*28
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(10, 12, kernel_size=(3,3))
        self.fc1 = nn.Linear(12 * 5 * 5, 10)
        self.fc2 = nn.Linear(10, 1)
        # self.f = F.tanh
        self.f = F.leaky_relu

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 1* 10 * 24 *24
        out = self.f(out)
        out = F.max_pool2d(out, 2, 2)  # 1* 10 * 12 * 12
        out = self.conv2(out)  # 1* 12 * 10 * 10
        out = self.f(out)
        out = F.max_pool2d(out, 2, 2)  # 1* 12 * 5 * 5
        out = out.view(in_size, -1)  # 1 * 300
        out = self.fc1(out)  # 1 * 20
        out = self.f(out)
        out = self.fc2(out)  # 1 * 1
        out = F.sigmoid(out)
        return out


def train():
    # Model parameters
    print_interval = 1 # EPOCHS/20
    d_steps = 20*1
    g_steps = 20*1

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    # d_learning_rate = 1e-3*1
    # g_learning_rate = 1e-3*2
    # d_sgd_momentum = 0.9
    # g_sgd_momentum = 0.9
    # d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=d_sgd_momentum)
    # g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=g_sgd_momentum)
    d_optimizer = optim.Adam(D.parameters(),lr=2e-4)
    g_optimizer = optim.Adam(G.parameters(), lr=4e-3)
    # g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
    # d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)

    loss_func = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

    dfe, dre, ge = 10, 10, 10
    d_real_data, d_fake_data, g_fake_data = None, None, None

    for epoch in range(EPOCHS):
        D.train()
        G.train()
        # while dre > 0.68 or dfe > 0.68:
        if 1:
            for d_index in range(d_steps):
                # 1. Train D on real+fake
                D.zero_grad()

                #  1A: Train D on real
                d_real_data,target = next(iter(train_loader))
                d_real_data = d_real_data.to(DEVICE)
                d_real_decision = D(d_real_data)
                # d_real_error = criterion(d_real_decision, torch.ones([1]))  # ones = true
                d_real_error = loss_func(d_real_decision, torch.ones([BATCH_SIZE,1]).to(DEVICE))
                d_real_error.backward() # compute/store gradients, but don't change params

                #  1B: Train D on fake
                d_gen_input = G_sampler(BATCH_SIZE)
                d_gen_input = d_gen_input.to(DEVICE)
                d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
                d_fake_decision = D(d_fake_data)
                # d_fake_error = criterion(d_fake_decision, torch.zeros([1]))  # zeros = fake
                d_fake_error = loss_func(d_fake_decision, torch.zeros([BATCH_SIZE,1]).to(DEVICE))
                d_fake_error.backward()
                d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

                dre, dfe = d_real_error.data.tolist(), d_fake_error.data.tolist()# extract(d_real_error)[0], extract(d_fake_error)[0]

        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            gen_input = G_sampler(BATCH_SIZE)
            gen_input = gen_input.to(DEVICE)
            g_fake_data = G(gen_input)
            dg_fake_decision = D(g_fake_data)
            # g_error = criterion(dg_fake_decision, torch.ones([1]))  # Train G to pretend it's genuine
            g_error = loss_func(dg_fake_decision, torch.ones([BATCH_SIZE,1]).to(DEVICE))
            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters

            ge = g_error.data.tolist() #extract(g_error)[0]
            # dfe = loss_func(dg_fake_decision, torch.zeros([BATCH_SIZE,1]).to(DEVICE)).data.tolist()

        if epoch % print_interval == 0:
            print("Epoch %s: D (%s real_err, %s fake_err) G (%s err);" %
                  (epoch, dre, dfe, ge))
        if epoch % 1 == 0:
            print("Plotting the generated distribution...")
            img = g_fake_data.data[0,0,:,:].cpu()
            img[0, 0] = 1 if img.max() > 1 else img[0, 0]
            img[0, 1] = 10 if img.max() > 10 else img[0, 1]
            img[0, 2] = 100 if img.max() > 100 else img[0, 2]
            plt.clf()
            plt.imshow(img)
            plt.title('g_fake_data')
            plt.pause(0.01)

    if 1:
        plt.show()


train()

