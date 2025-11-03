import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
# prinfc('inputs:', inputs)
prinfc('targets:', targets)

#  --------------------- 数据加载器
train_ds = TensorDataset(inputs, targets)
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


#  --------------------- 模型
model = torch.nn.Linear(3, 2)
loss_func = F.mse_loss
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)


#  --------------------- 训练
def train(epochs):
    for epoch in range(epochs):
        for a, b in train_dl:
            preds = model(a)
            loss = loss_func(preds, b)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # prinfc('loss:', loss)


train(300)
preds = model(inputs)
prinfc('preds:', preds)


#  --------------------- 保存
PATH = '\pylinear2.pt'
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