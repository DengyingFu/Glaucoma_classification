import argparse
from pprint import pprint
import random
from copy import deepcopy

import torch
import torch.backends
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.models import vgg16
from torchvision.transforms import transforms
from tqdm import tqdm

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal import ModelWrapper

import numpy as np
import matplotlib.pyplot as plt
from model import resnet18
import torch.nn as nn
from baal.utils.plot_utils import make_animation_from_data
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import cv2
"""
Minimal example to use BaaL.
这份代码是模拟主动学习过程，每次添加的标注数据是从train数据集文件夹里面选取的。初始假设给定一部分训练集数据有标签，对另一部分假设不知道标签
一轮学习完成后对不知道标签的数据进行预测，选取不确定性高的数据变成标注数据，加入训练
"""
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=2, type=int)  # 主动学习循环次数
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--initial_pool", default=5, type=int)
    parser.add_argument("--query_size", default=1, type=int)  # 训练完一轮要标记的数据量
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--heuristic", default=['bald'], type=str)  # 启发式方法 使用不同的启发式方法进行对比
    parser.add_argument("--iterations", default=1, type=int)  # 蒙特卡洛采样次数
    parser.add_argument("--shuffle_prop", default=0.01, type=float)  # 添加噪声
    parser.add_argument("--learning_epoch", default=2, type=int)
    return parser.parse_args()


def get_datasets(initial_pool):
    transform = transforms.Compose(
        [
            # transforms.CenterCrop((1650, 1650)), 根据不同的数据集，有些有黑边的需要中心裁剪 ORIGA
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "D:\青光眼辅助诊断系统\数据集\ACRIMA原始数据集\ACRIMA\\", "")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_ds = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                    transform=transform)
    # for i in range(len(train_ds)):
    #     img_pil = transforms.ToPILImage()(train_ds[i][0])
    #     plt.imshow(img_pil)
    #     plt.axis('off')  # 不显示坐标轴
    #     plt.show()
    test_set = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                    transform=test_transform)

    active_set = ActiveLearningDataset(train_ds, pool_specifics={"transform": test_transform})
    # We start labeling randomly.
    active_set.label_randomly(initial_pool)   #随机标记数据
    return active_set, test_set


active_set, test_set = get_datasets(10)


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    random.seed(1337)
    torch.manual_seed(1337)   #1337只是一个标志，每次运行时下面生成的随机数都是同一个
    Uncertain_th = 0
    fig, (ax, ax2) = plt.subplots(1,2)

    if not use_cuda:
        print("warning, the experiments would take ages to run on cpu")

    hyperparams = vars(args)   #返回字典对象
    print(hyperparams["heuristic"])
    for query_strategy in hyperparams["heuristic"]: #对每一种启发式方法查询策略进行主动学习，绘制性能曲线

        print("Start active learning Training: query_strategy is ", query_strategy, '\n')
        best_acc = 0
        performance = [] #用于记录每个查询策略的性能
        t_loss = []
        active_set, test_set = get_datasets(hyperparams["initial_pool"])
        if(query_strategy != 'bald'):
            reduction = 'mean'
        else:
            reduction = 'none'
        print('reduction on main is ', reduction)
        heuristic = get_heuristic(query_strategy, hyperparams["shuffle_prop"], reduction = reduction) #启发式算法bald，并添加噪声
        criterion = CrossEntropyLoss()
        model = vgg16(weights=None, num_classes=2)
        weights_path = "D:\青光眼辅助诊断系统\暑假工作\主动学习\\vgg16-397923af.pth"
        weights = torch.load(weights_path, map_location='cpu')
        weights = {k: v for k, v in weights.items() if "classifier.6" not in k}
        model.load_state_dict(weights, strict=False)

        # change dropout layer to MCDropout
        model = patch_module(model)

        if use_cuda:
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9)

        # Wraps the model into a usable API.
        model = ModelWrapper(model, criterion, replicate_in_memory=False)

        logs = {}
        logs["epoch"] = 0

        # for prediction we use a smaller batchsize
        # since it is slower
        active_loop = ActiveLearningLoop(
            active_set,
            model.predict_on_dataset,
            heuristic,  # 给定启发式算法
            hyperparams.get("query_size", 1),  # 每次标记的样本数
            batch_size=10,
            iterations=hyperparams["iterations"],  # 从后验分布采样次数
            use_cuda=use_cuda,
        )
        # We will reset the weights at each active learning step.
        init_weights = deepcopy(model.state_dict()) #每次主动学习前初始化权重

        labelling_progress = active_set.labelled.copy().astype(int)#将active_set对象的labelled属性复制为一个整数类型的数组labelling_progress。
        for _ in tqdm(range(args.epoch)):  #主动学习循环次数
            # Load the initial weights.
            model.load_state_dict(init_weights)
            model.train_on_dataset(  #这个函数可以返回一次主动循环的训练日志
                active_set,
                optimizer,
                hyperparams["batch_size"],
                hyperparams["learning_epoch"],  #训练的epoch数
                use_cuda,
            )

            # Validation!
            Uth, test_acc, test_loss = model.test_on_dataset(test_set, hyperparams["batch_size"], use_cuda, average_predictions=10)
            print('Uth-----------', Uth)
            performance.append(test_acc)
            t_loss.append(test_loss)
            should_continue = active_loop.step()    #执行主动学习步骤，根据启发式算法标记新样本
            # Keep track of progress
            labelling_progress += active_set._labelled.astype(np.uint16)#主动学习标记样本进度
            if not should_continue:
                break

            model_weight = model.state_dict()
            dataset = active_set.state_dict()
            if test_acc > best_acc:
                best_acc = test_acc
                Uncertain_th = Uth.cpu().numpy() #更新不确定性阈值
                print("!!!Save weights in "+query_strategy+str(_)+'al')
                print('Uncertain_th update to', Uncertain_th)
                torch.save({'model': model_weight, 'dataset': dataset, 'labelling_progress': labelling_progress},
                       'best_{}.pth'.format([query_strategy]))
            pprint(model.get_metrics())
        # 绘制性能曲线
        ax.plot(range(args.epoch), performance, label=query_strategy)#绘制一条曲线 横坐标为主动学习循环次数，纵坐标为test_acc
        ax2.plot(range(args.epoch), t_loss, label=query_strategy)
        # Make a feature extractor from our trained model.
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return torch.flatten(self.model.features(x), 1)

        features = FeatureExtractor(model.model)
        acc = []
        for x, y in DataLoader(active_set._dataset, batch_size=10):
            acc.append((features(x.cuda()).detach().cpu().numpy(), y.detach().cpu().numpy()))

        xs, ys = zip(*acc)
        # Compute t-SNE on the extracted features.
        tsne = TSNE(n_jobs=4, learning_rate='auto')
        transformed = tsne.fit_transform(np.vstack(xs))
        labels = np.concatenate(ys)
        print(labels.shape)
        # Create frames to animate the process.
        frames = make_animation_from_data(transformed, labels, labelling_progress, ["Positive", "Negative"])
        import imageio
        imageio.mimsave(query_strategy+'animation.gif', frames, duration=333.33)

    # 添加图例
    ax.legend()
    ax2.legend()
    # 添加横轴标签和纵轴标签
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Performance')
    ax2.set_xlabel('activate_iterations')
    ax2.set_ylabel('Loss')
    # 设置标题、坐标轴范围等其他设置
    ax.set_title('Performance Comparison')
    ax2.set_title('Loss in difference nums sample')
    ax.grid(True)
    ax2.grid(True)
    # 显示图形
    plt.show()


if __name__ == "__main__":
    main()
