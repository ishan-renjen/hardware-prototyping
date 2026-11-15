import logging
import threading
from concurrent import futures

import grpc
import numpy as np

import alphafold_blocks
import unary_sim_pb2
import unary_sim_pb2_grpc

log = logging.getLogger("gateway")

MAX_MSG = -1  # no gRPC limit (default is 4 MB); protobuf still caps one message at 2 GiB
DTYPES = {"f32": np.float32, "f16": np.float16, "s32": np.int32, "bool": np.bool_}
REV_DTYPES = {np.dtype(v): k for k, v in DTYPES.items()}

def decode(t):
    if t.dtype not in DTYPES:
        raise ValueError(f"unsupported dtype {t.dtype!r}")
    return np.frombuffer(t.data, dtype=DTYPES[t.dtype]).reshape(tuple(t.dims))

def encode(a):
    a = np.asarray(a)  # not ascontiguousarray: that turns 0-d results into shape (1,)
    if a.dtype not in REV_DTYPES:
        raise ValueError(f"unsupported output dtype {a.dtype}")
    return unary_sim_pb2.Tensor(dtype=REV_DTYPES[a.dtype], dims=list(a.shape), data=a.tobytes())

def module(signature, fn):
    """A hardware module described by its per-sample signature (NumPy gufunc syntax).

    fn computes one sample. Leading batch dimensions, as sent by a vmapped client
    call, are looped over one sample at a time; an operand sent without batch
    dimensions (e.g. weights) is reused for every sample.
    """
    return np.vectorize(fn, signature=signature)

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
        with self._lock:  # one hardware instance: one transaction at a time
            outs = self.modules[module](*inputs)
        return outs if isinstance(outs, tuple) else (outs,)

def tri_block(left_b, left_w, out_b, out_w, right_b, right_w, z):
    """A whole model block: two projections, a triangle contraction, an output projection.

    The client sends a block's weights first, in name order, then its input.
    """
    a = z @ left_w + left_b
    b = z @ right_w + right_b
    return np.einsum("ikc,jkc->ijc", a, b, optimize=True) @ out_w + out_b

def tri_block_torch(left_b, left_w, out_b, out_w, right_b, right_w, z):
    # torch.nn.Linear stores its weight as (out, in)
    return tri_block(left_b, left_w.T, out_b, out_w.T, right_b, right_w.T, z)

#JAX device ids starting from 0
COMPONENTS = {
    0: Component(0, "adder", {"jit_add_one": lambda x: x + 1}),
    1: Component(1, "scaler", {"jit_times_two": lambda x: x * 2}),
    2: Component(2, "vector", {
        "jit_vec_add": module("(n),(n)->(n)", np.add),
        "jit_vec_mul": module("(n),(n)->(n)", np.multiply),
        "jit_vec_gt": module("(n),(n)->(n)", np.greater),
        "jit_dot": module("(n),(n)->()", np.dot),
        "jit_axpy": module("(),(n),(n)->(n)", lambda a, x, y: a * x + y),
        "jit_vec_sum": module("(n)->()", lambda x: x.sum(dtype=x.dtype)),
        "jit_matvec": module("(n,m),(m)->(n)", np.matmul),
    }),
    3: Component(3, "trimul", {
        "jit_tri_mul": lambda a, b: np.einsum("ikc,jkc->ijc", a, b, optimize=True),
        "jit_tri_block": tri_block,
        "jit_tri_block_torch": tri_block_torch,
    }),
    # NumPy reference models of AlphaFold 2 blocks; replace each with the hardware model.
    4: Component(4, "alphafold", {
        "jit_transition": alphafold_blocks.transition,
        "jit_triangle_multiplication_outgoing": alphafold_blocks.triangle_multiplication("ikc,jkc->ijc"),
        "jit_triangle_multiplication_incoming": alphafold_blocks.triangle_multiplication("kjc,kic->ijc"),
        "jit_outer_product_mean": alphafold_blocks.outer_product_mean,
    }),
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
        try:
            outputs = comp.run(request.module, inputs)
        except Exception as e:  # wrong arity or operand shapes for this module
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          f"{request.module} on device {comp.id} rejected its inputs: {e}")
        try:
            encoded = [encode(o) for o in outputs]
        except ValueError as e:
            context.abort(grpc.StatusCode.INTERNAL, f"{request.module} on device {comp.id}: {e}")
        return unary_sim_pb2.ExecuteResponse(outputs=encoded, device=request.device, seq=request.seq)

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