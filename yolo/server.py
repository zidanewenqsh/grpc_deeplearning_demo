import grpc
import yolov5_pb2
import yolov5_pb2_grpc
from concurrent import futures
class YoloV5Service(yolov5_pb2_grpc.YoloV5ServiceServicer):
    def DetectObject(self, request, context):
        # 进行Yolov5图像检测（这里只是一个示例）
        image_data = request.image_data
        # ...（加载模型、执行检测）

        # 假设detection_results是检测结果
        detection_results = []

        return yolov5_pb2.Detections(detections=detection_results)

if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    yolov5_pb2_grpc.add_YoloV5ServiceServicer_to_server(YoloV5Service(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
