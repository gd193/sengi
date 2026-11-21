"""Register native query dependencies on a caller-owned Warp kernel."""
from threading import RLock
from typing import Any, Protocol
from weakref import WeakKeyDictionary

import warp as wp

from sengi.warp.native import merge_native_options


_registrations = WeakKeyDictionary()
_lock = RLock()


class DeviceQuery(Protocol):
    @property
    def _device_kernel(self) -> Any: ...


def configure_query_kernel(kernel: Any, *queries: DeviceQuery) -> Any:
    """Merge query build requirements before compiling a custom Warp kernel.

    Pass every query used by the module, including indirect calls. A new
    registration replaces the previous query requirements, if any.
    Existing include paths, dependencies, preambles and compiler flags survive.
    This performs host setup only: no compilation, launch or data transfer.
    """
    if not isinstance(kernel, wp.Kernel):
        raise TypeError("kernel must be a decorated Warp kernel")
    if not queries:
        raise ValueError("provide at least one query specialization")
    with _lock:
        module = kernel.module
        options = wp.get_module_options(module)
        own = options.get("extra_build_options")
        previous = _registrations.get(module)
        if previous is not None:
            original, registered = previous
            if own is registered:
                own = original
        requirements = wp.ModuleBuildOptions()
        unroll = options.get("max_unroll", 16)
        for query in queries:
            source = wp.get_module_options(query._device_kernel.module)
            requirements = merge_native_options(requirements, source["extra_build_options"])
            unroll = max(unroll, source.get("max_unroll", 16))
        build = requirements if own is None else own.merged(requirements)
        wp.set_module_options({"extra_build_options": build, "max_unroll": unroll}, module=module)
        _registrations[module] = own, build
    return kernel
