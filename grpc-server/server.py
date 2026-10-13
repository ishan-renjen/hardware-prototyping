import logging
import threading
from concurrent import futures

import grpc
import numpy as np

import unary_sim_pb2
import unary_sim_pb2_grpc

log = logging.getLogger("gateway")

MAX_MSG = 256 * 1024 * 1024  # raise gRPC's 4 MB default
DTYPES = {"f32": np.float32, "f16": np.float16, "s32": np.int32}
REV_DTYPES = {np.dtype(v): k for k, v in DTYPES.items()}

def decode(t):
    if t.dtype not in DTYPES:
        raise ValueError(f"unsupported dtype {t.dtype!r}")
    return np.frombuffer(t.data, dtype=DTYPES[t.dtype]).reshape(tuple(t.dims))

def encode(a):
    a = np.ascontiguousarray(a)
    return unary_sim_pb2.Tensor(dtype=REV_DTYPES[a.dtype], dims=list(a.shape), data=a.tobytes())

class Component:
    def __init__(self, id, kind, modules):
        self.id = id
        self.kind = kind
        self.modules = modules
        self._lock = threading.Lock()
        self._expected_seq = 0

    def check_seq(self, seq):
        with self._lock:
            if seq == 0:
                if self._expected_seq != 0:
                    log.info("device %d: seq reset to 0 (client restarted)", self.id)
                self._expected_seq = 1
                return
            if seq != self._expected_seq:
                log.warning("device %d: expected seq %d, got %d", self.id, self._expected_seq, seq)
            self._expected_seq = max(self._expected_seq, seq + 1)

    def run(self, module, inputs):
        outs = self.modules[module](*inputs)
        return outs if isinstance(outs, tuple) else (outs,)

#JAX device ids starting from 0
COMPONENTS = {
    0: Component(0, "adder", {"jit_add_one": lambda x: x + 1}),
    1: Component(1, "scaler", {"jit_times_two": lambda x: x * 2}),
}

class Gateway(unary_sim_pb2_grpc.SimulatorServicer):
    def ListDevices(self, request, context):
        return unary_sim_pb2.ListDevicesResponse(devices=[
            unary_sim_pb2.DeviceInfo(id=c.id, kind=c.kind, modules=sorted(c.modules))
            for c in COMPONENTS.values()])

    def Execute(self, request, context):
        comp = COMPONENTS.get(request.device)
        if comp is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"no device {request.device}")
        if request.module not in comp.modules:
            context.abort(grpc.StatusCode.UNIMPLEMENTED,
                          f"device {comp.id} ({comp.kind}) does not implement {request.module}")
        try:
            inputs = [decode(t) for t in request.inputs]
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        comp.check_seq(request.seq)
        log.debug("device %d seq %d: %s", comp.id, request.seq, request.module)
        outputs = comp.run(request.module, inputs)
        return unary_sim_pb2.ExecuteResponse(
            outputs=[encode(o) for o in outputs], device=request.device, seq=request.seq)

def serve(port=50051):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        options=[("grpc.max_receive_message_length", MAX_MSG),
                 ("grpc.max_send_message_length", MAX_MSG)])
    unary_sim_pb2_grpc.add_SimulatorServicer_to_server(Gateway(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("gateway listening on :%d with %d device(s)", port, len(COMPONENTS))
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()