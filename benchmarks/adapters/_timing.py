"""Resident Warp launches timed with CUDA graph events."""

from dataclasses import dataclass

import jax
import numpy as np
import warp as wp


@dataclass
class Launch:
    kernel: object
    inputs: list
    outputs: list
    size: int
    device: object
    block_dim: int

    def __call__(self):
        wp.launch(
            self.kernel,
            self.size,
            inputs=self.inputs,
            outputs=self.outputs,
            device=self.device,
            block_dim=self.block_dim,
        )


def bind_kernel(kernel, inputs, size, device, block_dim, shapes=None):
    parameters = kernel.adj.args
    output_types = parameters[len(inputs) :]
    output_shapes = shapes or ((size,),) * len(output_types)
    outputs = [
        wp.empty(shape, dtype=arg.type.dtype, device=device)
        for shape, arg in zip(output_shapes, output_types, strict=True)
    ]
    resident = []
    for value, arg in zip(inputs, parameters[: len(inputs)], strict=True):
        if isinstance(value, jax.Array):
            value = wp.from_jax(value, dtype=arg.type.dtype)
        elif isinstance(value, np.ndarray):
            value = wp.array(value, dtype=arg.type.dtype, device=device)
        if isinstance(value, wp.array) and value.device != device:
            raise ValueError(
                f"kernel input {arg.label} is on {value.device}, expected {device}"
            )
        resident.append(value)
    return Launch(kernel, resident, outputs, size, device, block_dim)


def measure(launch, options):
    device = launch.device
    if not device.is_cuda:
        raise ValueError("resident query timing requires a CUDA GPU")
    samples = []
    with wp.ScopedDevice(device):
        for _ in range(max(1, options.warmups)):
            launch()
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(options.launches):
                launch()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        start, stop = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        for _ in range(options.repeats):
            wp.record_event(start)
            wp.capture_launch(capture.graph)
            wp.record_event(stop)
            wp.synchronize_event(stop)
            samples.append(wp.get_event_elapsed_time(start, stop) / options.launches)
    return tuple(array.numpy() for array in launch.outputs), samples


def describe(launch, options):
    device = launch.device
    return {
        "device": str(device),
        "device_model": device.name,
        "timing_scope": "resident kernel",
        "instrumentation": "CUDA graph events",
        "block_dim": launch.block_dim,
        "launches": options.launches,
        "repeats": options.repeats,
        "warmups": max(1, options.warmups),
    }


def dtypes(arrays):
    return ", ".join(sorted({str(array.dtype) for array in arrays.values()}))
