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

    def derive(
            self, x: float = None, y: float = None, dx: float = None, dy: float = None,
            z: float = None, pathlength: float = None) -> 'Ray':
        '''
        Return a modified copy.

        Use this to modify some parameters while keeping others as-is
        '''
        return Ray(
            x=x if x is not None else self.x,
            y=y if y is not None else self.y,
            dx=dx if dx is not None else self.dx,
            dy=dy if dy is not None else self.dy,
            z=z if z is not None else self.z,
            pathlength=pathlength if pathlength is not None else self.pathlength,
            _one=self._one,
        )


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
