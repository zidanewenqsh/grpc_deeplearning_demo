import grpc
from concurrent import futures
import resnet_pb2
import resnet_pb2_grpc
import torch
from utils import load_model, load_labels, predict, preprocess_image, postprocess
# 导入你的ResNet模型和图像处理库，例如TensorFlow或PyTorch
# def init():
# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和标签
model = load_model(device=device)
labels = load_labels("imagenet_classes.txt")  # 请将路径替换为实际标签文件的路径

# 图像预处理
# input_batch = preprocess_image("cat01.jpg")  # 请将路径替换为实际图像文件的路径

# 进行预测
# probabilities = predict(model, input_batch, device)

# 后处理
# postprocess(probabilities, labels)
class ResNetService(resnet_pb2_grpc.ResNetServiceServicer):
    def ClassifyImage(self, request, context):
        image_data = request.image_data
        input_batch = preprocess_image(image_data) 
        probabilities = predict(model, input_batch, device)
        class_name, probability = postprocess(probabilities, labels)
        # 在这里添加图像预处理和ResNet模型推理的代码
        # 假设你得到了以下结果
        # class_name = "Dog"
        # probability = 0.95

        return resnet_pb2.ClassificationResult(class_name=class_name, probability=probability)

if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    resnet_pb2_grpc.add_ResNetServiceServicer_to_server(ResNetService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

