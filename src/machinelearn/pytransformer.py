# https://blog.csdn.net/weixin_50295745/article/details/127554480
import torch
import torchvision
from torch import nn
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from einops import rearrange
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchstat import stat
# from torchsummary import summary
import numpy as np
from src.utils.util import get_parameter_number, show_accuracy


#  --------------------- 超参数
BATCH_SIZE = 100
LEARNING_RATE = 0.01

TEST_BATCH_SIZE = 100


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
# 残差模块，放在每个前馈网络和注意力之后
class Residual(nn.Module):  # 通过连接层或者补充，保证fn输出和x是同维度的
    def __init__(self, fn):  # 带function参数的Module，都是嵌套的Module
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# layernorm归一化,放在多头注意力层和激活函数层。用绝对位置编码的BERT，layernorm用来自身通道归一化
class PreNorm(nn.Module):  # 先归一化，再用function作用。
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 三维的用dim，四维用[C,H,W]
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 放置多头注意力后，因为在于多头注意力使用的矩阵乘法为线性变换，后面跟上由全连 接网络构成的FeedForward增加非线性结构
class FeedForward(nn.Module):  # 非线性前馈，保持dim维不变
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


# 多头注意力层，多个自注意力连起来。使用qkv计算
class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

# Residual残差模块，放在每个前馈网络和注意力之后；
# Norm归一化,放在多头注意力层和激活函数层，用绝对位置编码的BERT，layernorm用来自身通道归一化；
# FeedForward放置多头注意力后，因为在于多头注意力使用的矩阵乘法为线性变换，后面跟上由全连接网络构成的FeedForward增加非线性结构；
# Attention多头注意力层，多个自注意力连起来，使用qkv计算。
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])  # ModuleList套ModuleList
        for _ in range(depth):  # 叠加Attention块
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


# 将图像切割成一个个图像块,组成序列化的数据输入Transformer执行图像分类任务。
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, '报错：图像没有被patch_size完美分割'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2  # (P**2 C)：一个patch展平为向量后实际的长度

        self.patch_size = patch_size
        # 维度看起来比较奇怪，dim,num_patches+1就可以解决问题
        # 还要加个1维度，还要把别的反过来，是为了适应b（batch）维度，进行广播
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # +1是为了适应cls_token
        self.patch_to_embedding = nn.Linear(patch_dim, dim)  # 将patch_dim（原图）经过embedding后得到dim维的嵌入向量
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size
        # print('init', img.shape)

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)  # 将H W C 转化成 N (P P C)
        # print('rearrange', x.shape)
        x = self.patch_to_embedding(x)  # 将(PPC)通过Embedding转化成一维embedding，这里的patch_to_embedding
        # print('patch_embedding', x.shape)
        # 到这里，一张图片就和nlp里的一个句子以同样的形式输入Transformer中
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)  # batch长度不确定，cat没有广播机制，所以要expand
        # print('cls_tokens', cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)  # 将类别信息接入embedding，0维是样本，1维是patch
        # print('cat cls', x.shape)
        # print('pos_embedding', self.pos_embedding.shape)
        x += self.pos_embedding  # +有广播机制，所以不需要expand
        x = self.transformer(x, mask)  # 送入encoder
        # print('after transformer', x.shape)
        x = self.to_cls_token(x[:, 0])  # 取出class对应的token，用Identity占位
        # print('cls_token', x.shape)
        y = self.mlp_head(x)  # 送入mlp分类器
        # print('mlp_head', y.shape)
        return y


'''
patch大小为 7x7（对于 28x28 图像，这意味着每个图像 4x4=16 个patch）、10 个可能的目标类别（0 到 9）和 1 个颜色通道（因为图像是灰度）。
在网络参数方面，使用了 64 个单元的维度，6 个Transformer 块的深度，8 个 Transformer 头，MLP 使用 128 维度。
'''
model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1, dim=64, depth=6, heads=8, mlp_dim=128)
loss_func = F.nll_loss
optimizer = optim.Adam(model.parameters(), lr=0.003)  # 优化器

#  --------------------- 训练
accuracies = []
loss_history = []


def evaluate(epochs, epoch):
    model.eval()

    total_samples = len(valid_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in valid_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = loss_func(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    loss_history.append(total_loss / total_samples)
    accuracies.append(correct_samples / total_samples)
    print("Epoch : [{}/{}], Loss:{}, {}:{}".format(epoch + 1, epochs, total_loss / total_samples, 'accuracy',
                                                   correct_samples / total_samples))


def train(epochs):
    for epoch in range(epochs):
        temp_time = time.time()
        model.train()

        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = F.log_softmax(model(data), dim=1)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

        # print("Epoch : [{}/{}], SpendTime:{}".format(epoch + 1, epochs, (time.time() - temp_time)))
        evaluate(epochs, epoch)


torch.manual_seed(42)


# stat(model, (3, 224, 224))
# summary(model, (3, 224, 224))
get_parameter_number(model)

print('训练前：')
evaluate(1, 0)

print('训练后：')
train(2)

show_accuracy(accuracies)


#  --------------------- 保存
PATH = '\pytransformer.pt'
torch.save(model, PATH)

#  --------------------- 加载
model_run = torch.load(PATH)
