"""Host contracts for the four statically specialized ray modes."""
from sengi.bvh.device_tree import DeviceTree
from typing import Generic, Literal, TypeVar, overload
import jax
from jax.typing import ArrayLike
import warp as wp
from sengi.queries.domain import Ray, RayOps
from sengi.warp.launch import BoundQuery
from sengi.warp.contracts import PrimitiveSpec, WarpFunction, DeviceArray, DeviceArray2D
from sengi.bvh.index import IndexData, IndexSpec

H = TypeVar("H")
D = TypeVar("D")
I = TypeVar("I")
P = TypeVar("P")
S = TypeVar("S")
Mode = TypeVar("Mode", Literal["first"], Literal["any"], Literal["count"], Literal["all"])
RayMode = Literal["first", "any", "count", "all"]

class _RayDevice:
    @overload
    def __get__(self, obj: RayQuery[H, D, I, P, Literal["first"]], objtype: type[object] | None = ...) -> WarpFunction[[P, P, float, float, int, DeviceTree, D, DeviceArray[float], DeviceArray[int], DeviceArray[int], DeviceArray[int], int], None]: ...
    @overload
    def __get__(self, obj: RayQuery[H, D, I, P, Literal["all"]], objtype: type[object] | None = ...) -> WarpFunction[[P, P, float, float, int, DeviceTree, D, DeviceArray2D[float], DeviceArray2D[int], DeviceArray[int], DeviceArray[int], int], None]: ...
    @overload
    def __get__(self, obj: RayQuery[H, D, I, P, Literal["any"]] | RayQuery[H, D, I, P, Literal["count"]], objtype: type[object] | None = ...) -> WarpFunction[[P, P, float, float, int, DeviceTree, D, DeviceArray[int], DeviceArray[int], int], None]: ...

class RayQuery(BoundQuery, Generic[H, D, I, P, Mode]):
    mode: Mode
    exclusion: bool
    device: _RayDevice
    @overload
    def __call__(self: RayQuery[H, D, I, P, Literal["first"]], data: IndexData[H], origin: ArrayLike, direction: ArrayLike, max_t: ArrayLike = ..., *, min_t: ArrayLike = ..., exclude_face: ArrayLike | None = ..., max_hits: None = ...) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ...
    @overload
    def __call__(self: RayQuery[H, D, I, P, Literal["any"]], data: IndexData[H], origin: ArrayLike, direction: ArrayLike, max_t: ArrayLike = ..., *, min_t: ArrayLike = ..., exclude_face: ArrayLike | None = ..., max_hits: None = ...) -> tuple[jax.Array, jax.Array]: ...
    @overload
    def __call__(self: RayQuery[H, D, I, P, Literal["count"]], data: IndexData[H], origin: ArrayLike, direction: ArrayLike, max_t: ArrayLike = ..., *, min_t: ArrayLike = ..., exclude_face: ArrayLike | None = ..., max_hits: None = ...) -> tuple[jax.Array, jax.Array]: ...
    @overload
    def __call__(self: RayQuery[H, D, I, P, Literal["all"]], data: IndexData[H], origin: ArrayLike, direction: ArrayLike, max_t: ArrayLike = ..., *, min_t: ArrayLike = ..., exclude_face: ArrayLike | None = ..., max_hits: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, I, P]], mode: Mode = "first", *, ops: RayOps[D, S], exclude: bool = ..., stack_capacity: int = ..., block_dim: int = ...) -> RayQuery[H, D, I, P, Mode]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, I, P]], mode: Mode = "first", *, ops: None = ..., exclude: bool = ..., stack_capacity: int = ..., block_dim: int = ...) -> RayQuery[H, D, I, P, Mode]: ...

ray_box: WarpFunction[[Ray, wp.vec3, wp.vec3, float, float, int], wp.vec2]
# Native vector overloads accept both dimensions.
valid_ray: WarpFunction[[wp.vec2 | wp.vec3, wp.vec2 | wp.vec3, float, float, int, int, bool, bool], bool]
