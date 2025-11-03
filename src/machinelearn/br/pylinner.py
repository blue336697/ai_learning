import torch
import numpy as np


def prinfc(s1, s2):
    print(s1)
    print(s2)
    print(" ")


#  --------------------- 数据
inputs = np.asarray([
    [73, 67, 43],
    [91, 88, 64],
    [87, 134, 58],
    [102, 43, 37],
    [69, 96, 70]
], dtype='float32')

targets = np.asarray([
    [56, 70],
    [81, 101],
    [119, 133],
    [22, 37],
    [103, 119]
], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
prinfc('训练前产量:', targets)

#  --------------------- 模型
weights = torch.randn(2, 3, requires_grad=True)
biases = torch.randn(2, requires_grad=True)
prinfc('训练前weights:', weights)


def model(x):
    return x @ weights.t() + biases


def loss_func(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


#  --------------------- 训练
for i in range(1):
    preds = model(inputs)  # 模型预测
    loss = loss_func(preds, targets)  # 计算损失
    loss.backward()  # 计算梯度

    with torch.no_grad():
        weights -= weights.grad * 1e-5  # 根据梯度优化权重
        biases -= biases.grad * 1e-5  # 根据梯度优化偏差
        weights.grad.zero_()  # 重置权重梯度
        biases.grad.zero_()  # 重置偏差梯度


prinfc('训练后weights:', weights)

preds = model(inputs)
prinfc('训练后预测:', preds)

#  --------------------- 保存
PATH = "\pylinear.pt"
torch.save(model, PATH)

#  --------------------- 加载
model_run = torch.load(PATH)

#  --------------------- 生产推理
inputs_run = np.asarray([
    [71, 85, 22],
    [121, 55, 96]
], dtype='float32')

inputs_run = torch.from_numpy(inputs_run)
pred_run = model_run(inputs_run)
prinfc('pred_run:', pred_run)
