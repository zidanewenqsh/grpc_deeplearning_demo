import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json

def load_model(model_name='resnet18', device='cpu'):
    """加载预训练模型"""
    model = getattr(models, model_name)(pretrained=True)
    model.eval()
    model = model.to(device)
    return model

def load_labels(label_path):
    """从文本文件加载标签"""
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def preprocess_image(image_path):
    """图像预处理"""
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor.unsqueeze(0)

def predict(model, input_batch, device='cpu'):
    """前向传播进行预测"""
    input_batch = input_batch.to(device)
    with torch.no_grad():
        output = model(input_batch)
    return torch.nn.functional.softmax(output[0], dim=0)

def postprocess(probabilities, labels):
    """结果后处理"""
    # print(probabilities.shape)
    # print(labels.shape)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(labels[top5_catid[i]], top5_prob[i].item())

if __name__ == "__main__":
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和标签
    model = load_model(device=device)
    labels = load_labels("imagenet_classes.txt")  # 请将路径替换为实际标签文件的路径

    # 图像预处理
    input_batch = preprocess_image("cat01.jpg")  # 请将路径替换为实际图像文件的路径

    # 进行预测
    probabilities = predict(model, input_batch, device)

    # 后处理
    postprocess(probabilities, labels)
