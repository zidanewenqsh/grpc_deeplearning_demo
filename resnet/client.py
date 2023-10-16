
import grpc
import resnet_pb2
import resnet_pb2_grpc
import sys
def load_image_as_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()
def run(imgpath):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = resnet_pb2_grpc.ResNetServiceStub(channel)
        # image_data = ...  # 加载和编码你的图像数据
        image_data = load_image_as_bytes(imgpath)
        response = stub.ClassifyImage(resnet_pb2.Image(image_data=image_data))
        print(f"Class: {response.class_name}, Probability: {response.probability}")

if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print("Usage: python3 client.py <image_path>")
        sys.exit()
    imgpath = sys.argv[1]
    run(imgpath)
