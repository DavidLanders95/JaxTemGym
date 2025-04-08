import jax_dataclasses as jdc
import jax.numpy as jnp
from numpy.typing import NDArray
from typing import Tuple
from collections.abc import Collection
from . import (
    PositiveFloat,
    Degrees,
)


@jdc.pytree_dataclass
class Ray:
    x: float
    y: float
    dx: float
    dy: float
    # Important to also include here for differentiation to work
    # Part of the matrix to allow constant offsets.
    # Don't set this as a user, has to be value 1!
    _one: float = 1.

    def modify(self, x=None, y=None, dx=None, dy=None) -> "Ray":
        loc = locals()
        params = {}
        for key in ('x', 'y', 'dx', 'dy'):
            ll = loc.get(key, None)
            if ll is None:
                ll = getattr(self, key)
            params[key] = ll
        return Ray(**params, _one=self._one)


@jdc.pytree_dataclass
class RayMatrix:
    matrix: jnp.ndarray

    @classmethod
    def from_rays(cls, rays: Collection[Ray]) -> "RayMatrix":
        vecs = tuple(jnp.array((r.x, r.y, r.dx, r.dy, r._one)) for r in rays)
        return cls(matrix=jnp.stack(vecs, axis=0))

    def to_rays(self) -> Collection[Ray]:
        res = tuple(
            Ray(x=v[0], y=v[1], z=v[2], dx=v[3], dy=v[4], _one=v[5])
            for v in self.matrix
        )
        return res


def unwrap(ray: Ray) -> jnp.ndarray:
    return RayMatrix.from_rays((ray, )).matrix[0]


def wrap(vec: jnp.ndarray) -> Ray:
    rays = RayMatrix(
        matrix=jnp.stack((vec, ), axis=0)
    )
    return rays.to_rays()[0]


@jdc.pytree_dataclass
class OldRay:
    z: float
    matrix: jnp.ndarray  # Shape (5,) vector [x, y, dx, dy, 1]
    pathlength: float

    @property
    def x(self):
        return self.matrix[..., 0]

    @property
    def y(self):
        return self.matrix[..., 1]

    @property
    def dx(self):
        return self.matrix[..., 2]

    @property
    def dy(self):
        return self.matrix[..., 3]


@jdc.pytree_dataclass
class GaussianRay(Ray):
    w0x: float = 1.0
    Rx: float = 0.0
    w0y: float = 1.0
    Ry: float = 0.0


def propagate(distance, ray: Ray):
    x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

    new_x = x + dx * distance
    new_y = y + dy * distance

    pathlength = ray.pathlength + distance * jnp.sqrt(1 + dx ** 2 + dy ** 2)

    Ray = ray_matrix(new_x, new_y, dx, dy,
                    ray.z + distance, pathlength)
    return Ray


def ray_matrix(x, y, dx, dy,
               z, pathlength):

    new_matrix = jnp.array([x, y, dx, dy, jnp.ones_like(x)]).T  # Doesnt work if all values have 0 shape

    return Ray(
        matrix=new_matrix,
        z=z,
        pathlength=pathlength,
    )
