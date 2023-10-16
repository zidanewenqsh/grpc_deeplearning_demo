import grpc
import yolov5_pb2
import yolov5_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = yolov5_pb2_grpc.YoloV5ServiceStub(channel)
        response = stub.DetectObject(yolov5_pb2.Image(image_data=image_data))
        
        # 输出检测结果
        for detection in response.detections:
            print(f"Class: {detection.class_name}, Confidence: {detection.confidence}")

if __name__ == '__main__':
    run()
