import jax_dataclasses as jdc
import jax.numpy as jnp
from jax.numpy import ndarray as NDArray
from typing import (
    Tuple
)

from .ray import Ray, propagate, ray_matrix
from .utils import R2P, P2R
from . import (
    Degrees,
)
from typing_extensions import TypeAlias

Radians: TypeAlias = jnp.float64  # type: ignore

@jdc.pytree_dataclass
class Lens:
    z: float
    focal_length: float

    def step(self, ray: Ray):
        f = self.focal_length

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy

        pathlength = ray.pathlength - (x ** 2 + y ** 2) / (2 * f)

        Ray = ray_matrix(x, y, new_dx, new_dy,
                        ray.z, ray.amplitude,
                        pathlength, ray.wavelength,
                        ray.blocked)
        return Ray


@jdc.pytree_dataclass
class ThickLens:
    z_po: float
    z_pi: float
    focal_length: float

    def step(self, ray: Ray):
        f = self.focal_length

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy

        pathlength = ray.pathlength - (x ** 2 + y ** 2) / (2 * f)

        new_z = ray.z - (self.z_po - self.z_pi)
        Ray = ray_matrix(x, y, new_dx, new_dy,
                        new_z, ray.amplitude,
                        pathlength, ray.wavelength,
                        ray.blocked)
        return Ray

    @property
    def z(self):
        return self.z_po

@jdc.pytree_dataclass
class Descanner:
    z: float
    descan_error: Tuple[float, float, float, float]  # Error in the scan position pos_x, y, tilt_x, y
    offset_x: float
    offset_y: float

    def step(self, ray: Ray):
        offset_x, offset_y = self.offset_x, self.offset_y

        (descan_error_xx, descan_error_xy, descan_error_yx, descan_error_yy,
         descan_error_dxx, descan_error_dxy, descan_error_dyx, descan_error_dyy) = self.descan_error
        
        descan_error_xx = 1.0 + descan_error_xx
        descan_error_yy = 1.0 + descan_error_yy
        
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_x = x * descan_error_xx + descan_error_xy * y + offset_x
        new_y = y * descan_error_yy + descan_error_yx * x + offset_y

        new_dx = dx + x * descan_error_dxx + y * descan_error_dxy
        new_dy = dy + y * descan_error_dyy + x * descan_error_dyx
        
        pathlength = ray.pathlength - (offset_x * x) - (offset_y * y)

        Ray = ray_matrix(new_x, new_y, new_dx, new_dy,
                         ray.z, ray.amplitude,
                         pathlength, ray.wavelength,
                         ray.blocked)
        return Ray


@jdc.pytree_dataclass
class Deflector:
    z: float
    def_x: float
    def_y: float

    def step(self, ray: Ray):

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        new_dx = dx + self.def_x
        new_dy = dy + self.def_y

        pathlength = ray.pathlength + dx * x + dy * y

        Ray = ray_matrix(x, y, new_dx, new_dy,
                        ray.z, ray.amplitude,
                        pathlength, ray.wavelength,
                        ray.blocked)
        return Ray

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
                        ray.z, ray.amplitude,
                        pathlength, ray.wavelength,
                        ray.blocked)
        return Ray

@jdc.pytree_dataclass
class DoubleDeflector:
    z: float
    first: Deflector
    second: Deflector

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
    scan_rotation: jdc.Static[Degrees]
    scan_step: jdc.Static[float]
    scan_shape: jdc.Static[Tuple[int, int]]
    center: jdc.Static[Tuple[float, float]]= (0., 0.)
    coords: jdc.Static[NDArray] = jdc.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "coords", self.get_coords())

    def step(self, ray: Ray):
        Ray = ray_matrix(ray.x, ray.y, ray.dx, ray.dy,
                        ray.z, ray.amplitude,
                        ray.pathlength, ray.wavelength,
                        ray.blocked)
        return Ray
    
    def get_coords(self):

        centre_x, centre_y = self.center
        scan_shape_y, scan_shape_x = self.scan_shape
        scan_step_y, scan_step_x = self.scan_step
        image_size_y = scan_shape_y * scan_step_y
        image_size_x = scan_shape_x * scan_step_x
        shape_y, shape_x = self.scan_shape

        y_image = jnp.linspace(-image_size_y / 2,
                               image_size_y / 2 - scan_step_y,
                               shape_y, endpoint=True) + centre_y
        
        x_image = jnp.linspace(-image_size_x / 2,
                               image_size_x / 2 - scan_step_x,
                               shape_x, endpoint=True) + centre_x


        y, x = jnp.meshgrid(y_image, x_image, indexing='ij')

        scan_rotation_rad = jnp.deg2rad(self.scan_rotation)

        y_rot = jnp.cos(scan_rotation_rad) * y - jnp.sin(scan_rotation_rad) * x
        x_rot = jnp.sin(scan_rotation_rad) * y + jnp.cos(scan_rotation_rad) * x

        r = jnp.stack((y_rot, x_rot), axis=-1).reshape(-1, 2)

        return r
    

    def scan_position(self, yx: Tuple[int, int]) -> Tuple[float, float]:
        y, x = yx
        # Get the scan position in physical units
        scan_step_y, scan_step_x = self.scan_step
        sy, sx = self.scan_shape
        scan_position_x = (x - sx / 2.) * scan_step_x
        scan_position_y = (y - sy / 2.) * scan_step_y

        scan_rotation_rad = jnp.deg2rad(self.scan_rotation)
        if scan_rotation_rad != 0.:
            pos_r, pos_a = R2P(scan_position_x + scan_position_y * 1j)
            pos_c = P2R(pos_r, pos_a + scan_rotation_rad)
            scan_position_y, scan_position_x = pos_c.imag, pos_c.real
        return (scan_position_y, scan_position_x)

    

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
    pixel_size: jdc.Static[float]
    shape: jdc.Static[Tuple[int, int]]
    center: jdc.Static[Tuple[float, float]] = (0., 0.)
    coords: jdc.Static[NDArray] = jdc.field(init=False)
    rotation: Degrees = 0.
    
    def __post_init__(self):
        object.__setattr__(self, "coords", self.get_coords())

    def step(self, ray: Ray):
        return ray

    def set_center_px(self, center_px: Tuple[int, int]):
        """
        For the desired image center in pixels (after any flip / rotation)
        set the image center in the physical coordinates of the microscope

        The continuous coordinate can be set directly on detector.center
        """
        iy, ix = center_px
        sy, sx = self.shape
        cont_y = (iy - sy // 2) * self.pixel_size
        cont_x = (ix - sx // 2) * self.pixel_size
        if self.flip_y:
            cont_y = -1 * cont_y
        mag, angle = R2P(cont_x + 1j * cont_y)
        coord: complex = P2R(mag, angle + jnp.deg2rad(self.rotation))
        self.center = coord.imag, coord.real

    def get_coords(self):
        centre_x, centre_y = self.center
        shape_y, shape_x = self.shape
        pixel_size = self.pixel_size
        image_size_y = shape_y * pixel_size
        image_size_x = shape_x * pixel_size

        y_lin = jnp.linspace(-image_size_y / 2,
                       image_size_y / 2 - pixel_size,
                       shape_y, endpoint=True) + centre_y

        x_lin = jnp.linspace(-image_size_x / 2,
                       image_size_x / 2 - pixel_size,
                       shape_x, endpoint=True) + centre_x

        y, x = jnp.meshgrid(y_lin, x_lin, indexing='ij')

        det_rotation_rad = jnp.deg2rad(self.rotation)

        y_rot = jnp.cos(det_rotation_rad) * y - jnp.sin(det_rotation_rad) * x
        x_rot = jnp.sin(det_rotation_rad) * y + jnp.cos(det_rotation_rad) * x

        r = jnp.stack((y_rot, x_rot), axis=-1).reshape(-1, 2)

        return r