from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchstat import stat
# from torchsummary import summary
import time
from src.utils.util import get_parameter_number, show_accuracy


#  --------------------- 超参数
BATCH_SIZE = 100
LEARNING_RATE = 0.001
NUM_OF_HIDDEN_UNITS = 256

TEST_BATCH_SIZE = 1000


#  --------------------- 数据
# ds = MNIST("e:/mnist", download=True, train=True, transform=trans.ToTensor())
# image, label = ds[0]
# plt.imshow(image[0], cmap="gray")
# plt.show()


#  --------------------- 数据加载器
def get_dataloader(train, batch_size=BATCH_SIZE):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])  # mean和std的形状与通道数相同

    ds = MNIST(root="e:/mnist", train=train, transform=transform_fn)

    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return data_loader


train_loader = get_dataloader(train=True)
valid_loader = get_dataloader(train=False, batch_size=TEST_BATCH_SIZE)


#  --------------------- 模型
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()  # 继承
        self.fc1 = nn.Linear(1 * 28 * 28, NUM_OF_HIDDEN_UNITS)  # 参数是input和output的feature
        self.fc2 = nn.Linear(NUM_OF_HIDDEN_UNITS, 10)

    def forward(self, input):
        # 1.进行形状的修改
        x = input.view([-1, 1 * 28 * 28])  # -1表示根据形状自动调整，也可以改为input.size(0)
        # 2.进行全连接的操作
        x = self.fc1(x)
        # 3.激活函数的处理
        x = F.relu(x)  # 形状没有变化
        # 4.输出层
        output = self.fc2(x)
        return F.log_softmax(output, dim=-1)


model = MnistModel()
loss_func = F.nll_loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#  --------------------- 训练
def accuracy(a, b):
    _, preds = torch.max(a, dim=1)
    return torch.sum(preds == b).item() / len(preds)


accuracies = []

def loss_batch(image, label, opt=None, metric=None):
    preds = model(image)
    loss = loss_func(preds, label)

    if (opt is not None):
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None
    if (metric is not None):
        metric_result = metric(preds, label)

    return loss.item(), len(image), metric_result

def evaluate(epochs, epoch):
    with torch.no_grad():
        results = [loss_batch(image, label, metric=accuracy)
                   for image, label in valid_loader]
        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        total_loss = np.sum(np.multiply(losses, nums))
        avg_loss = total_loss / total
        total_metric = np.sum(np.multiply(metrics, nums))
        avg_metric = total_metric / total

    accuracies.append(avg_metric)
    print("Epoch : [{}/{}], Loss:{}, {}:{}".format(epoch + 1, epochs, avg_loss, 'accuracy', avg_metric))


def train(epochs):  # epoch表示几轮
    for epoch in range(epochs):
        temp_time = time.time()

        for idx, (image, label) in enumerate(train_loader):  # idx表示data_loader中的第几个数据，元组是data_loader的数据
            loss_batch(image, label, opt=optimizer)

        print("Epoch : [{}/{}], SpendTime:{}".format(epoch + 1, epochs, (time.time() - temp_time)))

        evaluate(epochs, epoch)


# stat(model, (3, 224, 224))
# summary(model, (3, 224, 224))
get_parameter_number(model)

print('训练前：')
evaluate(1, 0)

print('训练后：')
train(2)

show_accuracy(accuracies)


#  --------------------- 保存
PATH = "\pydnn.pt"
torch.save(model, PATH)


#  --------------------- 加载
model_run = torch.load(PATH)

#  --------------------- 生产推理
test_ds = MNIST("e:/mnist", train=False, transform=ToTensor())


def predict_image(image):
    x = image.unsqueeze(0)
    y = model_run(x)
    _, pred = torch.max(y, dim=1)
    return pred[0].item()

print('生产推理：')
for i in range(30):
    test_image, test_label = test_ds[i]
    # plt.imshow(test_image[0], cmap="gray")
    # plt.show()

    print("Lable:", test_label, ", Predict:", predict_image(test_image))
