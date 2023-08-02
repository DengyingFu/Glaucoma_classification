import numpy as np
import torch
import torchvision
import torchvision.transforms as T

import matplotlib.pyplot as plt
from cs231n.data_utils import load_imagenet_val
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from PIL import Image

# 辅助函数


def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)


def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X


# 加载预训练模型
# Download and load the pretrained SqueezeNet model.
model = torchvision.models.squeezenet1_1(pretrained=True)

# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False


# 加载一些图片
X, y, class_names = load_imagenet_val(num=5)

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X[i])
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()



def compute_saliency_maps(X, y, model):
    """
    X表示图片, y表示分类结果, model表示使用的分类模型

    Input :
    - X : Input images : Tensor of shape (N, 3, H, W)
    - y : Label for X : LongTensor of shape (N,)
    - model : A pretrained CNN that will be used to computer the saliency map

    Return :
    - saliency : A Tensor of shape (N, H, W) giving the saliency maps for the input images
    """
    # 确保model是test模式
    model.eval()

    # 确保X是需要gradient
    X.requires_grad_()

    saliency = None

    logits = model.forward(X)
    logits = logits.gather(1, y.view(-1, 1)).squeeze()  # 得到正确分类
    logits.backward(torch.FloatTensor([1., 1., 1., 1., 1.]))  # 只计算正确分类部分的loss

    saliency = abs(X.grad.data)  # 返回X的梯度绝对值大小
    saliency, _ = torch.max(saliency, dim=1)

    return saliency.squeeze()


def show_saliency_maps(X, y):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()


show_saliency_maps(X, y)

