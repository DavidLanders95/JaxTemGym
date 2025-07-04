import jax_dataclasses as jdc
import jax.numpy as jnp


@jdc.pytree_dataclass
class Ray:
    x: float
    y: float
    dx: float
    dy: float
    z: float
    pathlength: float
    _one: float = 1.0


def propagate_dir_cosine(distance, ray):
    # This method implements propagation using direction cosines
    # and should be accurate to higher angles, but needs modification
    # to work with the rest of jaxgym transfer matrices
    N = jnp.sqrt(1 + ray.dx**2 + ray.dy**2)
    L = ray.dx / N
    M = ray.dy / N

    opl = distance * N

    new_ray = Ray(
        x=ray.x + L / N * distance,
        y=ray.y + M / N * distance,
        dx=ray.dx,
        dy=ray.dy,
        _one=1.0 * ray._one,
        z=ray.z * ray._one + distance,
        pathlength=ray.pathlength + opl,
    )
    return new_ray


def propagate(distance, ray):
    new_ray = Ray(
        x=ray.x + ray.dx * distance,
        y=ray.y + ray.dy * distance,
        dx=ray.dx,
        dy=ray.dy,
        _one=ray._one,
        z=ray.z + distance,
        pathlength=ray.pathlength + distance,
    )
    return new_ray
