import grpc
from concurrent import futures
from futur_grpc import Pose, Empty, PoseEstimateServicer, add_PoseEstimateServicer_to_server

class EchoServicer(PoseEstimateServicer):
    def TrackStream(self, request_iterator, context):
        for i, request in enumerate(request_iterator):
            print(f"[server] got frame {i}, width={request.width}, height={request.height}")
            yield Pose(translation=[0.0, 0.0, float(i)], rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1], success=True)

    def Reset(self, request, context):
        print("[server] reset")
        return Empty()

    def SeedPose(self, request, context):
        print(f"[server] seed pose translation={list(request.translation)}")
        return Empty()

server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
add_PoseEstimateServicer_to_server(EchoServicer(), server)
server.add_insecure_port('[::]:50051')
server.start()
print("[server] listening on 50051")
server.wait_for_termination()