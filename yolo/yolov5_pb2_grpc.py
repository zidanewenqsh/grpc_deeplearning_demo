# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import yolov5_pb2 as yolov5__pb2


class YoloV5ServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.DetectObject = channel.unary_unary(
                '/yolov5.YoloV5Service/DetectObject',
                request_serializer=yolov5__pb2.Image.SerializeToString,
                response_deserializer=yolov5__pb2.Detections.FromString,
                )


class YoloV5ServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def DetectObject(self, request, context):
        """检测接口
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_YoloV5ServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'DetectObject': grpc.unary_unary_rpc_method_handler(
                    servicer.DetectObject,
                    request_deserializer=yolov5__pb2.Image.FromString,
                    response_serializer=yolov5__pb2.Detections.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'yolov5.YoloV5Service', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class YoloV5Service(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def DetectObject(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/yolov5.YoloV5Service/DetectObject',
            yolov5__pb2.Image.SerializeToString,
            yolov5__pb2.Detections.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
