
from pprint import pprint
import torch
import torch.backends
from torch import optim
from torch.nn import CrossEntropyLoss
from torchvision.models import vgg16
from torchvision.transforms import transforms
from tqdm import tqdm
from baal.bayesian.dropout import patch_module
from baal import ModelWrapper

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import json
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc



def main():
    temp = []
    all_predictions = []#保存所有样本预测标签
    all_labels = []#保存所有样本的真值
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    batch_size = 5
    # load image
    img_dir = ['Normal', 'Glaucoma']
    for a in range(2):
        imgs_root = "D:\青光眼辅助诊断系统\数据集\ACRIMA原始数据集\ACRIMA\\val\\"+img_dir[a]
        assert os.path.exists(imgs_root), "file: '{}' dose not exist.".format(imgs_root)
        # 读取指定文件夹下所有jpg图像路径
        img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r") as f:
            class_indict = json.load(f)

        criterion = CrossEntropyLoss()
        model = vgg16(weights=None, num_classes=2)

        # load model weights
        weights_path = "D:\青光眼辅助诊断系统\暑假工作\\best_['bald'].pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        # 加载权重
        checkpoint = torch.load(weights_path, map_location='cuda:0')
        model.load_state_dict(checkpoint['model'])

        # change dropout layer to MCDropout
        model = patch_module(model)
        if use_cuda:
            model.cuda()

        # Wraps the model into a usable API.
        model = ModelWrapper(model, criterion, replicate_in_memory=False)

        # Validation!
        with torch.no_grad():
            for ids in range(0, len(img_path_list) // batch_size):
                img_list = []
                for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                    if a == 0:
                        all_labels.append(1)
                    elif a == 1:
                        all_labels.append(0)
                    assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                    img = Image.open(img_path)
                    img = data_transform(img)
                    img_list.append(img)
                # batch img
                # 将img_list列表中的所有图像打包成一个batch
                batch_img = torch.stack(img_list, dim=0)
                predictions = model.predict_on_batch(batch_img.to('cuda'), iterations=10).cpu() #iterations是进行mcdropout的次数
                batch_size, nclass, n_iteration = predictions.shape
                result = torch.softmax(predictions, dim=1)
                one_predict = result if ids==0 else np.vstack((one_predict,result)) #将一类的所有预测结果放到all_predict
                # print('re_',result)
                # print('all',one_predict) #samples x classes x n_iteration
            temp = temp+one_predict.tolist() #将两类样本预测结果合并到temp列表 大小nums*classes*iters

    all_pres = torch.tensor(temp)
    mean_pro = torch.mean(all_pres, dim=2)  # 计算多次MCdropout的预测概率平均值 大小nums*class
    # print('mean_Pro', mean_pro) #
    max_values, max_indices = torch.max(mean_pro, dim=1)  # 由MCdropout得到的平均概率值取得的预测结果
    print(max_values.shape, max_indices.shape) #最大概率值
    all_predictions = np.concatenate((all_predictions, max_indices.numpy())) #存储所有样本预测结果到一个列表内 （0或1）
    all_labels = np.array(all_labels)  #所有样本的真实标签（0或1）
    print(all_labels.shape)
    print(all_predictions.shape)

    # 预测熵
    predic_entropy = -torch.sum(mean_pro * torch.log2(mean_pro), dim=1)  # 对维度1求和，即类别
    print('predic_entropy = ', predic_entropy)  # 每个样本有自己的预测熵
    # uncertainty threshold
    thresholds = np.linspace(0.1, 1, num=90)
    # 初始化一个列表，用于存储在不同阈值下的模型准确率
    accuracies = []
    for threshold in thresholds:
        low_uncertainty_indices = list(np.where(predic_entropy < threshold)) # 保留低于不确定性阈值的位置索引
        low_uncertainty_predicts = np.array(all_predictions)[low_uncertainty_indices] #得到满足阈值条件的样本
        low_uncertainty_labels = np.array(all_labels)[low_uncertainty_indices]#真值
        accuracy = np.mean(low_uncertainty_predicts == low_uncertainty_labels)
        accuracies.append(accuracy)
    # print(accuracies)
    thresholds = np.array(thresholds)
    accuracies = np.array(accuracies)

    # 绘制准确率曲线图
    plt.plot(thresholds, accuracies)
    plt.xlabel('Uncertainty Threshold')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs. Uncertainty Threshold')
    plt.grid(True)
    plt.show()

    accuracy = np.mean(all_predictions == all_labels)
    # 计算混淆矩阵  (未使用预测熵不确定性)                        G N
    confusion_matrix = np.zeros((2, 2))              #G
    for i in range(len(all_labels)):                 #N
        true_class = all_labels[i]
        predicted_class = all_predictions[i]
        confusion_matrix[int(true_class)][int(predicted_class)] += 1 #行表示真值，列表示预测值

    # 计算灵敏度（召回率）
    # print(confusion_matrix[0][0]) #GG
    # print(confusion_matrix[0][1])  #GN
    # print(confusion_matrix[1][0]) #NG
    # print(confusion_matrix[1][1]) #NN
    sensitivity = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    print("Accuracy: {:.2f}".format(accuracy))
    print("Sensitivity (Recall): {:.2f}".format(sensitivity))

    f1 = f1_score(all_labels, all_predictions)
    print("F1-Score:{:2f}".format(f1))
    arr = 1 - all_labels #因为前面用0当做正样本，但是计算roc面积要用1当正样本，转换一下
    fpr, tpr, thresholds = roc_curve(arr, np.array(mean_pro[:, 0]))#mean_pro 是nums*classes的
    roc_auc = auc(fpr, tpr)
    print("AUC2:{}".format(roc_auc))
    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # 绘制直方图
    data = result[0, 0, :].numpy()
    bins = np.linspace(0, 1, num=50)
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins)
    plt.show()

    #绘制混淆矩阵
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Glaucoma', 'Normal'])
    plt.yticks(tick_marks, ['Glaucoma', 'Normal'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='black')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()
