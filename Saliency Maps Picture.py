import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from model import Model
import torch

# #导入数据集
# data_set = torchvision.datasets.MNIST("./MNIST",train=False,transform=torchvision.transforms.ToTensor())
# img,target = data_set[9365]

img = Image.open("4.png")
Ts = torchvision.transforms.Compose(
    [torchvision.transforms.Grayscale(num_output_channels=1),   #转换为灰度图像
     torchvision.transforms.Resize((28,28)),
     torchvision.transforms.ToTensor()]
)

img= Ts(img)



target = 4

img = torch.reshape(img,[-1,1,28,28])


model = torch.load("model_4.pth")  #导入模型



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

    logits = model(X)
    logits = logits[0, y]
    logits.backward(torch.FloatTensor([1.]))  # 只计算正确分类部分的loss

    saliency = abs(X.grad.data)  # 返回X的梯度绝对值大小
    saliency, _ = torch.max(saliency, dim=1)

    return saliency.squeeze()


def show_saliency_maps(X, y):
    # Convert X and y from numpy arrays to Torch Tensors
    # Compute saliency map for the image in X

    y = torch.tensor([y])
    saliency = compute_saliency_maps(X, y, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency map together.
    saliency = saliency.numpy()


    plt.figure(figsize=(12, 5))

    T = torchvision.transforms.ToPILImage()
    X = torch.reshape(X,[1, 28, 28])
    X = T(X)


    plt.subplot(1, 2, 1)
    plt.imshow(X, cmap='gray')  # Assuming X is of shape (1, 1, H, W)
    plt.axis('off')
    plt.title("Digit {}".format(y))

    plt.subplot(1, 2, 2)
    plt.imshow(saliency, cmap='Greens')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


print(target)
show_saliency_maps(img, target)


