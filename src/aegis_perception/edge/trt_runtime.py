"""Static batch-one TensorRT 8.5+/10 inference without a PyTorch runtime dependency."""

import ctypes as ct
import ctypes.util
import glob
from pathlib import Path

import numpy as np


class Cuda:
    """Small synchronous CUDA runtime wrapper; copies are included in measured calls."""

    def __init__(self):
        candidates = [ctypes.util.find_library("cudart"), "libcudart.so"]
        candidates += glob.glob("/usr/local/cuda*/targets/aarch64-linux/lib/libcudart.so*")
        candidates += glob.glob("/usr/local/cuda*/lib64/libcudart.so*")
        self.lib = None
        for name in candidates:
            if not name:
                continue
            try:
                self.lib = ct.CDLL(name)
                break
            except OSError:
                continue
        if self.lib is None:
            raise RuntimeError(
                "CUDA runtime not found. Use the JetPack-provided CUDA installation."
            )
        signatures = {
            "cudaMalloc": ([ct.POINTER(ct.c_void_p), ct.c_size_t], ct.c_int),
            "cudaFree": ([ct.c_void_p], ct.c_int),
            "cudaMemcpy": ([ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int], ct.c_int),
            "cudaStreamCreate": ([ct.POINTER(ct.c_void_p)], ct.c_int),
            "cudaStreamDestroy": ([ct.c_void_p], ct.c_int),
            "cudaStreamSynchronize": ([ct.c_void_p], ct.c_int),
            "cudaGetDeviceCount": ([ct.POINTER(ct.c_int)], ct.c_int),
            "cudaMemGetInfo": ([ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)], ct.c_int),
            "cudaGetErrorString": ([ct.c_int], ct.c_char_p),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        count = ct.c_int()
        self.check(self.lib.cudaGetDeviceCount(ct.byref(count)))
        if count.value < 1:
            raise RuntimeError("No CUDA device available")

    def check(self, status):
        if status:
            raise RuntimeError(f"CUDA {status}: {self.lib.cudaGetErrorString(status).decode()}")

    def malloc(self, count):
        pointer = ct.c_void_p()
        self.check(self.lib.cudaMalloc(ct.byref(pointer), count))
        return pointer.value

    def free(self, pointer):
        if pointer:
            self.check(self.lib.cudaFree(pointer))

    def upload(self, pointer, array):
        self.check(self.lib.cudaMemcpy(pointer, array.ctypes.data, array.nbytes, 1))

    def download(self, array, pointer):
        self.check(self.lib.cudaMemcpy(array.ctypes.data, pointer, array.nbytes, 2))

    def memory(self):
        free, total = ct.c_size_t(), ct.c_size_t()
        self.check(self.lib.cudaMemGetInfo(ct.byref(free), ct.byref(total)))
        return {"free_bytes": free.value, "total_bytes": total.value}


class TrtSession:
    """One input, one output, fixed shapes, reusable allocations and checked execution."""

    def __init__(self, path):
        import tensorrt as trt

        self.cuda = Cuda()
        self.stream = ct.c_void_p()
        self.pointers = {}
        self.cuda.check(self.cuda.lib.cudaStreamCreate(ct.byref(self.stream)))
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            self.close()
            raise RuntimeError("Engine cannot be loaded; rebuild on this Jetson/TensorRT version")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            self.close()
            raise RuntimeError("Cannot allocate TensorRT execution context")
        self.named = hasattr(self.engine, "num_io_tensors") and hasattr(
            self.context, "execute_async_v3"
        )
        self.inputs, self.outputs, self.arrays = [], [], {}
        try:
            count = self.engine.num_io_tensors if self.named else self.engine.num_bindings
            self.bindings = [0] * count
            for i in range(count):
                name = (
                    self.engine.get_tensor_name(i)
                    if self.named
                    else self.engine.get_binding_name(i)
                )
                shape = tuple(
                    self.engine.get_tensor_shape(name)
                    if self.named
                    else self.engine.get_binding_shape(i)
                )
                dtype = (
                    self.engine.get_tensor_dtype(name)
                    if self.named
                    else self.engine.get_binding_dtype(i)
                )
                is_input = (
                    (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
                    if self.named
                    else self.engine.binding_is_input(i)
                )
                tensor_format = (
                    self.engine.get_tensor_format(name)
                    if self.named
                    else self.engine.get_binding_format(i)
                )
                if tensor_format != trt.TensorFormat.LINEAR:
                    raise ValueError("Only linear model I/O is supported; rebuild with this kit")
                if not shape or any(d <= 0 for d in shape):
                    raise ValueError("Only fixed positive batch-one shapes are supported")
                if dtype not in (trt.float32, trt.float16):
                    raise ValueError("Expected floating-point model I/O")
                array = np.empty(shape, dtype=np.float32 if dtype == trt.float32 else np.float16)
                self.arrays[name] = array
                self.pointers[name] = self.cuda.malloc(array.nbytes)
                self.bindings[i] = self.pointers[name]
                (self.inputs if is_input else self.outputs).append(name)
                if self.named and not self.context.set_tensor_address(name, self.pointers[name]):
                    raise RuntimeError(f"Could not bind {name}")
            if len(self.inputs) != 1 or len(self.outputs) != 1:
                raise ValueError("Expected exactly one input and one output")
        except BaseException:
            self.close()
            raise
        self.input_shape = self.arrays[self.inputs[0]].shape

    def run(self, x):
        name = self.inputs[0]
        array = self.arrays[name]
        x = np.asarray(x)
        if x.shape != array.shape or not np.isfinite(x).all():
            raise ValueError(f"Invalid input, expected {array.shape}")
        np.copyto(array, x, casting="same_kind")
        self.cuda.upload(self.pointers[name], array)
        ok = (
            self.context.execute_async_v3(self.stream.value)
            if self.named
            else self.context.execute_async_v2(self.bindings, self.stream.value)
        )
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        self.cuda.check(self.cuda.lib.cudaStreamSynchronize(self.stream))
        name = self.outputs[0]
        self.cuda.download(self.arrays[name], self.pointers[name])
        result = self.arrays[name].copy()
        if not np.isfinite(result).all():
            raise FloatingPointError("TensorRT produced NaN/Inf; use validated FP32 fallback")
        return result

    def close(self):
        # Release execution context before memory buffers.
        self.context = None
        for pointer in self.pointers.values():
            self.cuda.free(pointer)
        self.pointers.clear()
        if self.stream.value:
            self.cuda.check(self.cuda.lib.cudaStreamDestroy(self.stream))
            self.stream = ct.c_void_p()
        self.engine = None
        self.runtime = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
