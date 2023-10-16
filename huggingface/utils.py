import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
from io import BytesIO
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


def preprocess_image(image_data):
    """图像预处理"""
    # input_image = Image.open(image_path)
    input_image = Image.open(BytesIO(image_data))
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
    """结果后处理，返回最大概率和对应的分类标签"""
    max_prob, max_catid = torch.max(probabilities, dim=0)
    max_label = labels[max_catid]
    return max_label, max_prob.item()

# 示例代码
# 假设`probabilities`是一个1D张量，包含每个类别的概率值
# 假设`labels`是一个包含所有可能类别标签的列表
# probabilities = torch.tensor([0.1, 0.2, 0.7, 0.0])
# labels = ["cat", "dog", "bird", "fish"]

# 使用函数
# max_label, max_prob = postprocess(probabilities, labels)
# print(f"Max Label: {max_label}, Max Probability: {max_prob}")


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
