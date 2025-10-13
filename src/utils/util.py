import matplotlib.pyplot as plt

def get_parameter_number(model):
    print('模型参数：')
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})

def show_accuracy(accuracies):
    plt.plot(accuracies, "-x")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()