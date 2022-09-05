import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
torch.set_default_tensor_type(torch.DoubleTensor)

LOAD_MODEL = 1
SAVE_MODEL = 1
EPOCHS = 10
learning_rate = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1000
train_batch_num = 1000
test_batch_num = 100
train_set_size = train_batch_num * batch_size
test_set_size = test_batch_num * batch_size
maxrandom = 100

# 开始生成个样本
train_set = []
for m in range(train_batch_num):
    x = np.random.random((batch_size,1))*maxrandom #batch=1000
    y = np.random.random((batch_size,1))*maxrandom
    x_y = np.hstack((x,y))
    xy = x*y

    data = torch.from_numpy(x_y)
    target = torch.from_numpy(xy)
    train_set.append((data,target))

test_set = []
for m in range(test_batch_num):
    x = np.random.random((batch_size,1))*maxrandom
    y = np.random.random((batch_size,1))*maxrandom
    x_y = np.hstack((x,y))
    xy = x*y

    data = torch.from_numpy(x_y)
    target = torch.from_numpy(xy)
    test_set.append((data,target))


class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        in_size = x.size(0)
        out = self.fc1(x)  # 1 * 500
        out = F.leaky_relu(out)
        out = self.fc2(out)  # 1 * 10
        out = F.leaky_relu(out)
        return out


# 定义训练函数
train_losses = []
def train(model, device, train_set, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_set):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss
        if (batch_idx + 1) % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_set), 100. * batch_idx / len(train_set), loss.item()))
    train_loss /= len(train_set)
    train_losses.append(train_loss.item())
    if SAVE_MODEL:
        torch.save(model.state_dict(), './data/model.pth')
        torch.save(optimizer.state_dict(), './data/optimizer.pth')


# 定义测试函数
test_losses = []
def test(model, device, test_set):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_set:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target)  # 将一批的损失相加
    test_loss /= len(test_set)
    test_losses.append(test_loss.item())
    print("\nTest set: Average loss: {:.6f}\n".format(test_loss))


#生成模型和优化器
model = FCNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

if LOAD_MODEL:
    model_state_dict = torch.load('data/model.pth')
    optimizer_state_dict = torch.load('data/optimizer.pth')
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    print('model load successfully')


# 最后开始训练和测试
for epoch in range(1, EPOCHS + 1):
    train(model,  DEVICE, train_set, optimizer, epoch)
    test(model, DEVICE, test_set)
    # time.sleep(1)


show_size = 10
x = np.random.random((show_size,1))*maxrandom*1 #batch=1000
y = np.random.random((show_size,1))*maxrandom*1
x_y = np.hstack((x,y))
xy = x*y
data = torch.from_numpy(x_y)
target = torch.from_numpy(xy)
with torch.no_grad():
    output = model(data.to(DEVICE))
output = output.cpu().numpy()
for i in range(x.shape[0]):
    print('x:{} y:{} output:{} target:{}'.format(x[i],y[i],output[i],xy[i]))



plt.plot(train_losses)
plt.plot(test_losses)
plt.ylim(0,1.05*max(max(train_losses),max(test_losses)))
plt.legend(['train loss','test loss'])
plt.show()
