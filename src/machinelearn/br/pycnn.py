from torchvision.datasets import MNIST
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torchstat import stat
# from torchsummary import summary
import time
from src.utils.util import get_parameter_number, show_accuracy


#  --------------------- 超参数
BATCH_SIZE = 100
LEARNING_RATE = 0.01

TEST_BATCH_SIZE = 100

momentum = 0.5


#  --------------------- 数据加载器
def get_dataloader(train, batch_size=BATCH_SIZE):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])  # mean和std的形状与通道数相同

    ds = MNIST(root="d:/mnist", train=train, transform=transform_fn)

    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return data_loader
#
# def get_dataloader(train):
#     transform_fn = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
#     ds = MNIST(root="d:/mnist", train=train, transform=transform_fn, download=True)
#     return DataLoader(ds, batch_size=32, shuffle=True)

train_loader = get_dataloader(train=True)
valid_loader = get_dataloader(train=False, batch_size=TEST_BATCH_SIZE)


#  --------------------- 模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


model = Net()
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=momentum)  # lr学习率，momentum冲量


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


def train(epochs):
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
PATH = r"\pycnn.pt"
torch.save(model, PATH)

#  --------------------- 加载
model_run = torch.load(PATH)

#  --------------------- 生产推理
test_ds = MNIST("d:/mnist", train=False, transform=ToTensor())


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
