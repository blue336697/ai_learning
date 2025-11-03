import torch
import torchvision
import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torchstat import stat

from src.utils.util import get_parameter_number, show_accuracy


def prinfc(s1, s2):
    print(s1)
    print(s2)
    print(" ")


#  --------------------- 超参数

BATCH_SIZE = 100
LEARNING_RATE = 0.001

TEST_BATCH_SIZE = 1000


#  --------------------- 数据
# ds = MNIST("e:/mnist", download=True, train=True, transform=trans.ToTensor())
# image, label = ds[0]
# plt.imshow(image[0], cmap="gray")
# plt.show()


#  --------------------- 数据加载器
def get_dataloader(train, batch_size=BATCH_SIZE):
    transform_fn = trans.Compose([
        trans.ToTensor(),
        trans.Normalize(mean=(0.1307,), std=(0.3081,))
    ])  # mean和std的形状与通道数相同

    ds = MNIST(root="e:/mnist", train=train, transform=transform_fn)

    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return data_loader


train_loader = get_dataloader(train=True)
valid_loader = get_dataloader(train=False, batch_size=TEST_BATCH_SIZE)


# fig = plt.figure()
# for i in range(12):
#     plt.subplot(3, 4, i+1)
#     plt.tight_layout()
#     plt.imshow(valid_loader.dataset.train_data[i], cmap='gray', interpolation='none')
#     plt.title("Labels: {}".format(valid_loader.dataset.train_labels[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

#  --------------------- 模型
class MnistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28 * 28, 10)

    def forward(self, input):
        input = input.reshape(-1, 784)
        output = self.linear(input)
        return output


model = MnistModel()
loss_func = F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


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
        for image, label in train_loader:
            loss_batch(image, label, opt=optimizer)

        evaluate(epochs, epoch)


# stat(model, (3, 224, 224))
get_parameter_number(model)

print('训练前：')
evaluate(1, 0)

print('训练后：')
train(2)

show_accuracy(accuracies)

#  --------------------- 保存
PATH = '\pylogic.pt'
torch.save(model, PATH)

#  --------------------- 加载
model_run = torch.load(PATH)

#  --------------------- 生产推理
test_ds = MNIST("e:/mnist", train=False, transform=trans.ToTensor())


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
