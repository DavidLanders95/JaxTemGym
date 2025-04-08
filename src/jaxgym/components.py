from abc import ABC, abstractmethod

import jax
import jax_dataclasses as jdc
import jax.numpy as jnp
from typing import (
    Tuple, Callable
)

from .ray import Ray, RayMatrix, wrap, unwrap, propagate, ray_matrix
from .coordinate_transforms import apply_transformation, pixels_to_metres_transform
from . import (
    Degrees, Coords_XY, Scale_YX, Coords_YX, Pixels_YX, Shape_YX
)
from typing_extensions import TypeAlias

Radians: TypeAlias = jnp.float64  # type: ignore
EPS = 1e-12


class Component(ABC):
    @abstractmethod
    def step(self, ray: Ray) -> Ray:
        pass

    @abstractmethod
    def pathlength(self, ray: Ray) -> float:
        pass


def _naked_step(component: Component) -> Callable[[jnp.ndarray], jnp.ndarray]:
    '''
    Derive a function that maps a ray vector to a ray vector
    so that the derivative is nice and clean, without the classes etc.
    '''
    def wrapper(vec: jnp.ndarray) -> jnp.ndarray:
        ray = wrap(vec)
        res = component.step(ray)
        return unwrap(res)

    return wrapper


class LinearComponent(Component):
    def validate(self):
        wrapped = _naked_step(component=self)
        hessian = jax.jacobian(jax.jacobian(wrapped))
        # Just have to make sure we are not in a step-wise function or sth like that...
        # TODO take some shots at this to try make a naughty function pass
        sample = Ray(
            x=0., y=0., dx=0., dy=0.
        )
        hess = hessian(unwrap(sample))
        if jnp.any(hess != 0.):
            raise ValueError('Second derivative of step() is not zero: Not a linear function.')


@jdc.pytree_dataclass
class MatrixComponent:
    matrix: jnp.ndarray

    @classmethod
    def from_component(cls, component: LinearComponent) -> "MatrixComponent":
        # Make sure it is actually linear
        component.validate()
        # We calculate the Jacobian of a function
        # that maps ndarray to ndarray so that we get a clean matrix as a result
        # Since we make sure it is a linear function of the ray,
        # the Jacobian is the transfer matrix of the component
        jac = jax.jacobian(_naked_step(component=component))

        # Any ray will do
        sample = Ray(
            x=1., y=2., dx=4., dy=5.
        )
        # Get the Jacobian which is the transfer matrix
        mat = jac(unwrap(sample))
        # ...et voila, we can make a MatrixComponent from that
        # transfer matrix
        return cls(mat.T)

    def step(self, rays: RayMatrix) -> RayMatrix:
        return RayMatrix(rays.matrix @ self.matrix)


@jdc.pytree_dataclass
class Identity(LinearComponent):
    def step(self, ray: Ray) -> Ray:
        return ray

    def pathlength(self, ray):
        return 0


@jdc.pytree_dataclass
class FreeSpace(LinearComponent):
    length: float

    def step(self, ray: Ray) -> Ray:
        return ray.modify(
            x=ray.x + self.length * ray.dx,
            y=ray.y + self.length * ray.dy,
        )

    def pathlength(self, ray: Ray) -> float:
        return self.length * jnp.linalg.norm(
            jnp.array((1, ray.dx, ray.dy))
        )


@jdc.pytree_dataclass
class Lens(LinearComponent):
    focal_length: float

    def step(self, ray: Ray) -> Ray:
        return ray.modify(
            dx=ray.dx - ray.x / self.focal_length,
            dy=ray.dy - ray.y / self.focal_length
        )

    def pathlength(self, ray: Ray) -> float:
        return -(ray.x**2 + ray.y**2) / (2 * self.focal_length)


@jdc.pytree_dataclass
class ThickLens(Lens):
    thickness: float

    def step(self, ray: Ray):
        thin_ray = super().step(ray)
        return thin_ray.modify(
            z=thin_ray.z + self.thickness * thin_ray._one
        )


@jdc.pytree_dataclass
class Shifter(LinearComponent):
    offset_x: float
    offset_y: float

    def step(self, ray: Ray):
        return ray.modify(
            x=ray.x + self.offset_x * ray._one,
            y=ray.y + self.offset_y * ray._one,
        )

    def pathlength(self, ray):
        return -(self.offset_x * ray.x) - (self.offset_y * ray.y)


@jdc.pytree_dataclass
class Tilter(LinearComponent):
    tilt_x: float
    tilt_y: float

    def step(self, ray: Ray):
        return ray.modify(
            dx=ray.dx + self.tilt_x * ray._one,
            dy=ray.dy + self.tilt_y * ray._one,
        )

    def pathlength(self, ray):
        return 0.


@jdc.pytree_dataclass
class DescanError:
    xx: float = 0.
    xy: float = 0.
    yx: float = 0.
    yy: float = 0.
    dxx: float = 0.
    dxy: float = 0.
    dyx: float = 0.
    dyy: float = 0.


def scan_descan_triplet(offset_x: float, offset_y: float, descan_error: DescanError) \
        -> tuple[LinearComponent, LinearComponent, LinearComponent]:
    scan = Shifter(offset_x=offset_x, offset_y=offset_y)
    descan = Shifter(
        offset_x=-offset_x
        + offset_x * descan_error.xx
        + offset_y * descan_error.xy,
        offset_y=-offset_y
        + offset_x * descan_error.yx
        + offset_y * descan_error.yy,
    )
    tilt = Tilter(
        tilt_x=offset_x * descan_error.dxx + offset_y * descan_error.dxy,
        tilt_y=offset_x * descan_error.dyx + offset_y * descan_error.dyy,
    )
    return (scan, descan, tilt)


@jdc.pytree_dataclass
class Rotator:
    z: float
    angle: Degrees

    def step(self, ray: Ray):
            
        angle = jnp.deg2rad(self.angle)

        # Rotate the ray's position
        new_x = ray.x * jnp.cos(angle) - ray.y * jnp.sin(angle)
        new_y = ray.x * jnp.sin(angle) + ray.y * jnp.cos(angle)
        # Rotate the ray's slopes
        new_dx = ray.dx * jnp.cos(angle) - ray.dy * jnp.sin(angle)
        new_dy = ray.dx * jnp.sin(angle) + ray.dy * jnp.cos(angle)

        pathlength = ray.pathlength

        Ray = ray_matrix(new_x, new_y, new_dx, new_dy,
                        ray.z,
                        pathlength)
        return Ray


@jdc.pytree_dataclass
class DoubleDeflector:
    z: float
    first: Tilter
    second: Tilter

    def step(self, ray: Ray):
        ray = self.first.step(ray)
        z_step = self.second.z - self.first.z
        ray = propagate(z_step, ray)
        ray = self.second.step(ray)

        return ray


@jdc.pytree_dataclass
class InputPlane:
    z: float   

    def step(self, ray: Ray):
        return ray
    

@jdc.pytree_dataclass
class PointSource:
    z: float   
    semi_conv: float

    def step(self, ray: Ray):
        return ray
    

@jdc.pytree_dataclass
class ScanGrid:
    z: float
    scan_step: Scale_YX
    scan_shape: jdc.Static[Shape_YX]
    scan_rotation: Degrees
    scan_centre: Coords_XY = (0., 0.)
    metres_to_pixels_mat: jnp.ndarray = jdc.field(init=False)
    pixels_to_metres_mat: jnp.ndarray = jdc.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "metres_to_pixels_mat", self.get_metres_to_pixels_transform())
        object.__setattr__(self, "pixels_to_metres_mat", self.get_pixels_to_metres_transform())

    @property
    def coords(self) -> jnp.ndarray:
        return self.get_coords()


    def step(self, ray: Ray):
        return ray


    def get_coords(self) -> Coords_XY:

        scan_shape = self.scan_shape

        y_px = jnp.arange(scan_shape[0])
        x_px = jnp.arange(scan_shape[1])

        yy_px, xx_px = jnp.meshgrid(y_px, x_px, indexing='ij')

        yy_px = yy_px.ravel()
        xx_px = xx_px.ravel()

        scan_coords_x, scan_coords_y = self.pixels_to_metres((yy_px, xx_px))

        scan_coords_xy = jnp.stack((scan_coords_x, scan_coords_y), axis=-1).reshape(-1, 2)

        return scan_coords_xy
    

    def get_metres_to_pixels_transform(self) -> jnp.ndarray:
        centre = self.scan_centre
        step = self.scan_step
        shape = self.scan_shape
        rotation = self.scan_rotation
        pixels_to_metres_mat = pixels_to_metres_transform(centre, step, shape, False, rotation)
        metres_to_pixels_mat = jnp.linalg.inv(pixels_to_metres_mat)

        return metres_to_pixels_mat
    

    def get_pixels_to_metres_transform(self) -> jnp.ndarray:
        centre = self.scan_centre
        step = self.scan_step
        shape = self.scan_shape
        rotation = self.scan_rotation
        pixels_to_metres_mat = pixels_to_metres_transform(centre, step, shape, False, rotation)

        return pixels_to_metres_mat
     

    def metres_to_pixels(self, coords: Coords_XY) -> Pixels_YX:
        coords_x, coords_y = coords
        pixels_y, pixels_x = apply_transformation(coords_y, coords_x, self.metres_to_pixels_mat)
        pixels_y = jnp.round(pixels_y).astype(jnp.int32)       
        pixels_x = jnp.round(pixels_x).astype(jnp.int32)   

        return pixels_y, pixels_x


    def pixels_to_metres(self, pixels: Pixels_YX) -> Coords_XY:
        pixels_y, pixels_x = pixels
        metres_y, metres_x = apply_transformation(pixels_y, pixels_x, self.pixels_to_metres_mat)

        return metres_x, metres_y
    
    
@jdc.pytree_dataclass
class Aperture:
    z: float
    radius: float
    x: float = 0.
    y: float = 0.

    def step(self, ray: Ray):

        pos_x, pos_y, pos_dx, pos_dy = ray.x, ray.y, ray.dx, ray.dy
        distance = jnp.sqrt(
            (pos_x - self.x) ** 2 + (pos_y - self.y) ** 2
        )
        # This code evaluates to 1 if the ray is blocked already,
        # even if the new ray is inside the aperture,
        # evaluates to 1 if the ray was not blocked before and is now,
        # and evaluates to 0 if the ray was not blocked before and is NOT now.
        blocked = jnp.where(distance > self.radius, 1, ray.blocked)

        Ray = ray_matrix(pos_x, pos_y, pos_dx, pos_dy,
                        ray.z, ray.amplitude,
                        ray.pathlength, ray.wavelength,
                        blocked)
        return Ray


@jdc.pytree_dataclass
class Biprism:
    z: float
    offset: float = 0.
    rotation: Degrees = 0.
    deflection: float = 0.

    def step(
        self, ray: Ray,
    ) -> Ray:

        pos_x, pos_y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        deflection = self.deflection
        offset = self.offset
        rot = jnp.deg2rad(self.rotation)

        rays_v = jnp.array([pos_x, pos_y]).T

        biprism_loc_v = jnp.array([offset*jnp.cos(rot), offset*jnp.sin(rot)])

        biprism_v = jnp.array([-jnp.sin(rot), jnp.cos(rot)])
        biprism_v /= jnp.linalg.norm(biprism_v)

        rays_v_centred = rays_v - biprism_loc_v

        dot_product = jnp.dot(rays_v_centred, biprism_v) / jnp.dot(biprism_v, biprism_v)
        projection = jnp.outer(dot_product, biprism_v)

        rejection = rays_v_centred - projection
        rejection = rejection/jnp.linalg.norm(rejection, axis=1, keepdims=True)

        # If the ray position is located at [zero, zero], rejection_norm returns a nan,
        # so we convert it to a zero, zero.
        rejection = jnp.nan_to_num(rejection)

        xdeflection_mag = rejection[:, 0]
        ydeflection_mag = rejection[:, 1]

        new_dx = (dx + xdeflection_mag * deflection).squeeze()
        new_dy = (dy + ydeflection_mag * deflection).squeeze()

        pathlength = ray.pathlength + (
            xdeflection_mag * deflection * pos_x + ydeflection_mag * deflection * pos_y
        )

        Ray = ray_matrix(pos_x.squeeze(), pos_y.squeeze(), new_dx, new_dy,
                        ray.z, ray.amplitude,
                        pathlength, ray.wavelength,
                        ray.blocked)
        return Ray


@jdc.pytree_dataclass
class Detector:
    z: float
    det_pixel_size: Scale_YX
    det_shape: jdc.Static[Shape_YX]
    det_centre: Coords_XY = (0., 0.)
    det_rotation: Degrees = 0.
    flip_y: bool = False
    metres_to_pixels_mat: jnp.ndarray = jdc.field(init=False)
    pixels_to_metres_mat: jnp.ndarray = jdc.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "metres_to_pixels_mat", self.get_metres_to_pixels_transform())
        object.__setattr__(self, "pixels_to_metres_mat", self.get_pixels_to_metres_transform())


    @property
    def coords(self) -> jnp.ndarray:
        return self.get_coords()


    def step(self, ray: Ray):
        return ray


    def get_coords(self) -> Coords_XY:

        det_shape = self.det_shape

        y_px = jnp.arange(det_shape[0])
        x_px = jnp.arange(det_shape[1])

        yy_px, xx_px = jnp.meshgrid(y_px, x_px, indexing='ij')

        yy_px = yy_px.ravel()
        xx_px = xx_px.ravel()

        det_coords_x, det_coords_y = self.pixels_to_metres((yy_px, xx_px))

        det_coords_xy = jnp.stack((det_coords_x, det_coords_y), axis=-1).reshape(-1, 2)

        return det_coords_xy
    

    def get_metres_to_pixels_transform(self) -> jnp.ndarray:
        centre = self.det_centre
        step = self.det_pixel_size
        shape = self.det_shape
        rotation = self.det_rotation
        pixels_to_metres_mat = pixels_to_metres_transform(centre, step, shape, False, rotation)
        metres_to_pixels_mat = jnp.linalg.inv(pixels_to_metres_mat)

        return metres_to_pixels_mat
    
    def get_pixels_to_metres_transform(self) -> jnp.ndarray:
        centre = self.det_centre
        step = self.det_pixel_size
        shape = self.det_shape
        rotation = self.det_rotation
        pixels_to_metres_mat = pixels_to_metres_transform(centre, step, shape, False, rotation)

        return pixels_to_metres_mat
     

    def metres_to_pixels(self, coords: Coords_XY) -> Pixels_YX:
        coords_x, coords_y = coords
        pixels_y, pixels_x = apply_transformation(coords_y, coords_x, self.metres_to_pixels_mat)

        pixels_y = jnp.round(pixels_y).astype(jnp.int32)
        pixels_x = jnp.round(pixels_x).astype(jnp.int32)  

        return pixels_y, pixels_x


    def pixels_to_metres(self, pixels: Pixels_YX) -> Coords_XY:
        pixels_y, pixels_x = pixels
        metres_y, metres_x = apply_transformation(pixels_y, pixels_x, self.pixels_to_metres_mat)

        return metres_x, metres_y
