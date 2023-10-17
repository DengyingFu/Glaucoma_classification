import torch
import torch.backends
from torch.nn import CrossEntropyLoss
from torchvision.models import vgg16
from torchvision.transforms import transforms
from baal.bayesian.dropout import patch_module
from baal import ModelWrapper

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import json

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
import cv2


#数据预处理
use_cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(3 * [0.5], 3 * [0.5]),
    ]
)

img_path = "ACRIMA/train/Normal/Im134_ACRIMA.jpg"

assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img_init = Image.open(img_path)
# [N, C, H, W]
img = data_transform(img_init)
img = torch.unsqueeze(img, dim=0)
# 读取类别
# 读取类别
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)

criterion = CrossEntropyLoss()
model = vgg16(num_classes=2)

# load model weights
weights_path = "best_['bald'].pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
# 加载权重
checkpoint = torch.load(weights_path, map_location='cuda:0')
model.load_state_dict(checkpoint['model'])
model_grad = model

# dropout层 改成 MCDropout
model = patch_module(model)
if use_cuda:
    model.cuda()

# 包装模型
model = ModelWrapper(model, criterion, replicate_in_memory=False)

with torch.no_grad():
    predictions = model.predict_on_batch(img.to('cuda'), iterations=100).cpu()  # iterations是进行mcdropout的次数
    result = torch.softmax(predictions, dim=1)

    max_values, max_indices = torch.max(result, dim=1)  # 每个iteration的最大概率及其对应类别
    mean_pro = torch.mean(result, dim=2)  # 对iteration的值求均值

    max_mean_pro, max_mean_indices = torch.max(mean_pro, dim=1)  # 两个类分别的概率 0是Positive，1是Negative
    print(max_mean_pro, max_mean_indices)

for i in range(max_indices.size(1)):
    prediction_values = max_values[0, i]  # 维度0只有0，维度1有i次预测即i个
    prediction_indices = max_indices[0, i]
print("Prediction {}: Probabilistic = {:.4f}, Class = {}".format(i, prediction_values,
                                                                 class_indict[str(prediction_indices.numpy())]))
# 暂时以平均概率值来判别分类类别
print_res = "Probabilistic = {:.4f}, Class = {}".format(max_mean_pro[0],
                                                        class_indict[str(max_mean_indices[0].numpy())])
# 预测熵
mean_positive = torch.mean(result, dim=2)
# print(mean_positive)
predic_entropy = -torch.sum(mean_pro * torch.log2(mean_pro), dim=1)
print('predic_entropy = ', predic_entropy)
# 数据
data = result[0, 0, :].numpy()  # 第二维=0，因为第二维表示类别，0在我的模型中表示正类。即取0位置的概率值就表示越接近1就越有可能是正类
bins = np.linspace(0, 1, num=50)
# 绘制直方图
fig, ax = plt.subplots()
ax.hist(data, bins=bins)
plt.xlabel("Probablistic")
plt.ylabel("iterations")
plt.show()
# 不确定性过大或预测概率过小都拒绝给出预测
if predic_entropy > 0.5048 or max_mean_pro < 0.8:
    plt.imshow(img_init)
    plt.title("Can not give result")
    print("我们的模型对您的眼底照片预测结果产生较大的不确定性，不能给出可信的预测")

else:
    # 显示图形
    plt.imshow(img_init)
    plt.title(print_res)
    print(print_res)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img =img.to(device)
if max_mean_indices[0].item() == 1:
    targets = [ClassifierOutputTarget(0)] #括号内输出要分析的类别，即预测结果
else:
    targets = [ClassifierOutputTarget(1)]

targets = [ClassifierOutputTarget(1)]
target_layers = [model_grad.features[23]]
cam = GradCAM(model=model_grad, target_layers=target_layers, use_cuda=True)

cam_map = cam(input_tensor=img, targets=targets)[0] # 不加平滑

import torchcam
from torchcam.utils import overlay_mask

result = overlay_mask(img_init, Image.fromarray(cam_map), alpha=0.6) # alpha越小，原图越淡

plt.subplot(1, 2, 1)
plt.imshow(img_init)  # Assuming X is of shape (1, 1, H, W)
plt.axis('off')
plt.title("origin_pricture")

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.axis('off')
plt.title("grad_cam")
plt.show()
print("分类结果为：{}".format(class_indict[str(max_mean_indices[0].numpy())]))