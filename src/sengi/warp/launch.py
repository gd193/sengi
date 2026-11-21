"""Typed array entry points for factory-built Warp queries."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from hashlib import sha256
from pathlib import Path
from threading import RLock
import linecache
from itertools import count
from typing import Any

import warp as wp

from sengi.jax.ffi import batched_kernel
from sengi.warp.native import query_build_options
from sengi.warp.arrays import load_array, store_array
from sengi.bvh.device_tree import DeviceTree


_names = count()


@dataclass(frozen=True)
class BoundQuery:
    kernel: object
    primitive: object
    num_query_args: int
    num_outputs: int
    block_dim: int
    device: Any = field(kw_only=True)
    group_types: tuple = field(default=(), kw_only=True)
    components: tuple[tuple[str, str], ...] = field(default=(), kw_only=True)

    @property
    def _device_kernel(self):
        return self.kernel

    def describe(self) -> dict[str, object]:
        """Describe the selected host-side composition without compiling it."""
        return {
            "query": type(self).__name__,
            "primitive": getattr(self.primitive, "name", type(self.primitive).__name__),
            "block_dim": self.block_dim,
            "components": dict(self.components),
        }

    def launch(self, tree, geometry, shape, *args, output_shapes=None, metadata=()):
        tree.validate_geometry(geometry, self.primitive)
        if len(metadata) != len(self.group_types):
            raise ValueError("query metadata does not match its static schema")
        call = batched_kernel(self.kernel, self.num_query_args, self.num_outputs, self.block_dim, output_shapes)
        extra = tuple(a for value, schema in zip(metadata, self.group_types, strict=True)
                      for a in struct_arrays(value, schema))
        values = call(*args, *struct_arrays(tree, DeviceTree), *struct_arrays(geometry, self.primitive.data_type), *extra)
        return tuple(v.reshape(shape+v.shape[1:]) for v in values)


def array_kernel(name, arguments, body, bindings, *, build=None, max_unroll=None):
    """Materialize only the flat FFI signature; algorithms stay Warp functions."""
    declarations = ", ".join(f"{key}: T{i}" for i, (key, _) in enumerate(arguments))
    source = f"def {name}({declarations}):\n" + "\n".join("    " + line for line in body) + "\n"
    filename = f"<warp_bvh_{name}_{next(_names)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"__name__": __name__, "wp": wp, "load_array": load_array, "store_array": store_array, **bindings}
    namespace.update({f"T{i}": value for i, (_, value) in enumerate(arguments)})
    exec(compile(source, filename, "exec"), namespace)
    build = query_build_options() if build is None else build
    options = {"extra_build_options": build}
    if max_unroll is not None:
        options["max_unroll"] = max_unroll
    return wp.kernel(module="unique", enable_backward=False, module_options=options)(namespace[name])


def struct_arguments(prefix, struct_type):
    if any(not isinstance(var.type, wp.array) for var in struct_type.vars.values()):
        raise TypeError("payload struct fields must be Warp arrays")
    return [(f"{prefix}_{name}", var.type) for name, var in struct_type.vars.items()]


def struct_body(prefix, struct_type, symbol):
    return [f"{prefix} = {symbol}()", *(
        f"{prefix}.{name} = {prefix}_{name}" for name in struct_type.vars)]


def struct_arrays(value, struct_type):
    return tuple(getattr(value, name) for name in struct_type.vars)


def bind_query(query_type, name, primitive, inputs, outputs, body, bindings, block_dim, *,
               device, groups=(), native_providers=(), native_requirements=(), max_unroll=None, **options):
    schemas = (("tree", DeviceTree), ("geometry", primitive.data_type), *groups)
    arguments = [*inputs, *(arg for prefix, schema in schemas for arg in struct_arguments(prefix, schema)), *outputs]
    body = [*(line for prefix, schema in schemas for line in struct_body(prefix, schema, prefix+"_type")), *body]
    bindings = {**{prefix+"_type": schema for prefix, schema in schemas}, **bindings}
    build = query_build_options(primitive, *native_providers, requirements=native_requirements)
    kernel = array_kernel(name, arguments, ["i = wp.tid()", *body], bindings,
                          build=build, max_unroll=max_unroll)
    return query_type(kernel, primitive, len(inputs), len(outputs), block_dim,
                      device=device, group_types=tuple(schema for _, schema in groups), **options)


def cache_query(factory):
    """Reuse a specialization while the contents of its native headers match."""
    cache = {}
    lock = RLock()

    @wraps(factory)
    def bind(*args, **kwargs):
        key = (args, tuple(kwargs.items()))
        with lock:
            previous = cache.get(key)
            if previous is not None:
                query, dependencies, fingerprint = previous
                if _header_fingerprint(dependencies) == fingerprint:
                    return query
            query = factory(*args, **kwargs)
            build = wp.get_module_options(query._device_kernel.module)["extra_build_options"]
            dependencies = tuple(Path(path) for path in build.extra_build_dependencies)
            cache[key] = query, dependencies, _header_fingerprint(dependencies)
            return query

    return bind


def _header_fingerprint(dependencies):
    return tuple(sha256(path.read_bytes()).digest() for path in dependencies)
