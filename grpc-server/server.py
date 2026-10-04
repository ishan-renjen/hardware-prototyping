from concurrent import futures
import grpc
import numpy as np
import simulator_pb2, simulator_pb2_grpc

MAX_MSG = 256 * 1024 * 1024  #gRPC's 4 MB default
DTYPES = {"f32": np.float32, "f16": np.float16, "bf16": None, "s32": np.int32}

class StubSimulator(simulator_pb2_grpc.SimulatorServicer):
    def Execute(self, request, context):
        #stub - echo, can be changed later
        return simulator_pb2.ExecuteResponse(outputs=list(request.inputs))

def serve(port=50051):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[("grpc.max_receive_message_length", MAX_MSG), ("grpc.max_send_message_length", MAX_MSG)])
    simulator_pb2_grpc.add_SimulatorServicer_to_server(StubSimulator(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()