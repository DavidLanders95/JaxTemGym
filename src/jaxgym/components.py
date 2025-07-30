import jax_dataclasses as jdc
import jax.numpy as jnp

from .ray import Ray
from typing_extensions import TypeAlias
from .ode import solve_ode
from . import Degrees

Radians: TypeAlias = jnp.float64  # type: ignore
EPS = 1e-12


@jdc.pytree_dataclass
class Component:
    def __call__(self, ray: Ray) -> Ray:
        raise NotImplementedError()


@jdc.pytree_dataclass
class Sampling(Component):
    def __call__(self, ray: Ray):
        return ray.derive()


@jdc.pytree_dataclass
class FreeSpace(Component):
    thickness: float

    def __call__(self, ray: Ray):
        return ray.derive(
            x=ray.x + ray.dx * self.thickness,
            y=ray.y + ray.dy * self.thickness,
            z=ray.z + self.thickness,
            # Is this correct, through, for rays not parallel to the optical axis?
            pathlength=ray.pathlength + self.thickness,
        )


# TODO eliminate
def propagate(thickness: float, ray: Ray) -> Ray:
    return FreeSpace(thickness=thickness)(ray)


@jdc.pytree_dataclass
class Lens(Component):
    focal_length: float

    def __call__(self, ray: Ray):
        f = self.focal_length

        dx = -ray.x / f + ray.dx
        dy = -ray.y / f + ray.dy

        pathlength = ray.pathlength - (ray.x**2 + ray.y**2) / (2 * f)

        return ray.derive(dx=dx, dy=dy, pathlength=pathlength)


@jdc.pytree_dataclass
class ThickLens(Component):
    thickness: float
    focal_length: float

    def __call__(self, ray: Ray):
        f = self.focal_length

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy

        pathlength = ray.pathlength - (x**2 + y**2) / (2 * f)

        new_z = ray.z + self.thickness

        return ray.derive(
            dx=new_dx, dy=new_dy, pathlength=pathlength, z=new_z
        )




@jdc.pytree_dataclass
class Scanner(Component):
    scan_pos_x: float
    scan_pos_y: float
    scan_tilt_x: float = 0.
    scan_tilt_y: float = 0.

    def __call__(self, ray: Ray):
        """
        The traditional 5x5 linear ray transfer matrix of an optical system is
               [Axx, Axy, Bxx, Bxy, pos_offset_x],
               [Ayx, Ayy, Byx, Byy, pos_offset_y],
               [Cxx, Cxy, Dxx, Dxy, slope_offset_x],
               [Cyx, Cyy, Dyx, Dyy, slope_offset_y],
               [0.0, 0.0, 0.0, 0.0, 1.0],
        Since the Scanner is designed to only shift or tilt the entire incoming beam,
        with a certain error as a function of scan position, we write the 5th column
        of the ray transfer matrix, which is designed to describe an offset in shift or tilt,
        as a linear function of the scan position (spx, spy) (ignoring scan tilt for now):
        """
        return ray.derive(
            x=ray.x + self.scan_pos_x * ray._one,
            y=ray.y + self.scan_pos_y * ray._one,
            dx=ray.dx + self.scan_tilt_x * ray._one,
            dy=ray.dy + self.scan_tilt_y * ray._one,
        )


@jdc.pytree_dataclass
class DescanError:
    pxo_pxi: float = 0.0  # How position x output scales with respect to scan x position
    pxo_pyi: float = 0.0  # How position x output scales with respect to scan y position
    pyo_pxi: float = 0.0  # How position y output scales with respect to scan x position
    pyo_pyi: float = 0.0  # How position y output scales with respect to scan y position
    sxo_pxi: float = 0.0  # How slope x output scales with respect to scan x position
    sxo_pyi: float = 0.0  # How slope x output scales with respect to scan y position
    syo_pxi: float = 0.0  # How slope y output scales with respect to scan x position
    syo_pyi: float = 0.0  # How slope y output scales with respect to scan y position
    offpxi: float = 0.0  # Constant additive error in x position
    offpyi: float = 0.0  # Constant additive error in y position
    offsxi: float = 0.0  # Constant additive error in x slope
    offsyi: float = 0.0  # Constant additive error in y slope


@jdc.pytree_dataclass
class Descanner(Component):
    # Will be applied in reverse
    scan_pos_x: float
    scan_pos_y: float
    scan_tilt_x: float = 0.
    scan_tilt_y: float = 0.
    descan_error: DescanError = DescanError()

    def __call__(self, ray: Ray):
        """
        The traditional 5x5 linear ray transfer matrix of an optical system is
               [Axx, Axy, Bxx, Bxy, pos_offset_x],
               [Ayx, Ayy, Byx, Byy, pos_offset_y],
               [Cxx, Cxy, Dxx, Dxy, slope_offset_x],
               [Cyx, Cyy, Dyx, Dyy, slope_offset_y],
               [0.0, 0.0, 0.0, 0.0, 1.0],
        Since the Descanner is designed to only shift or tilt the entire incoming beam,
        with a certain error as a function of scan position, we write the 5th column
        of the ray transfer matrix, which is designed to describe an offset in shift or tilt,
        as a linear function of the scan position (spx, spy) (ignoring scan tilt for now):
        Thus -
            pos_offset_x(spx, spy) = pxo_pxi * spx + pxo_pyi * spy + offpxi
            pos_offset_y(spx, spy) = pyo_pxi * spx + pyo_pyi * spy + offpyi
            slope_offset_x(spx, spy) = sxo_pxi * spx + sxo_pyi * spy + offsxi
            slope_offset_y(spx, spy) = syo_pxi * spx + syo_pyi * spy + offsyi
        which can be represented as another 5x5 transfer matrix that is used to populate
        the 5th column of the ray transfer matrix of the optical system. The jacobian call
        in jaxgym will return the complete 5x5 ray transfer matrix of the optical system
        with the total descan error included in the 5th column.
        """
        de = self.descan_error
        sp_x, sp_y = self.scan_pos_x, self.scan_pos_y
        st_x, st_y = self.scan_tilt_x, self.scan_tilt_y

        return ray.derive(
            x=ray.x + (
                sp_x * de.pxo_pxi
                + sp_y * de.pxo_pyi
                + de.offpxi
                - sp_x
            ) * ray._one,
            y=ray.y + (
                sp_x * de.pyo_pxi
                + sp_y * de.pyo_pyi
                + de.offpyi
                - sp_y
            ) * ray._one,
            dx=ray.dx + (
                sp_x * de.sxo_pxi
                + sp_y * de.sxo_pyi
                + de.offsxi
                - st_x
            ) * ray._one,
            dy=ray.dy + (
                sp_x * de.syo_pxi
                + sp_y * de.syo_pyi
                + de.offsyi
                - st_y
            ) * ray._one
        )


@jdc.pytree_dataclass
class ODE:
    # TODO eliminate Z
    z: float
    z_end: float
    phi_lambda: callable
    E_lambda: callable

    def step(self, ray: Ray) -> Ray:
        in_state = jnp.array([ray.x, ray.y, ray.dx, ray.dy, ray.pathlength])

        z_start = self.z
        z_end = self.z_end

        u0 = self.phi_lambda(0.0, 0.0, z_start).astype(jnp.float64)

        out_state, out_z = solve_ode(
            in_state, z_start, z_end, self.phi_lambda, self.E_lambda, u0
        )

        x, y, dx, dy, opl = out_state

        return ray.derive(x=x, y=y, dx=dx, dy=dy, pathlength=opl, z=out_z)


@jdc.pytree_dataclass
class Deflector:
    def_x: float
    def_y: float

    def step(self, ray: Ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        new_dx = dx + self.def_x
        new_dy = dy + self.def_y

        pathlength = ray.pathlength + dx * x + dy * y

        return Ray(
            x=x,
            y=y,
            dx=new_dx,
            dy=new_dy,
            _one=ray._one,
            pathlength=pathlength,
            z=ray.z,
        )


@jdc.pytree_dataclass
class Rotator:
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

        return Ray(
            x=new_x,
            y=new_y,
            dx=new_dx,
            dy=new_dy,
            _one=ray._one,
            pathlength=pathlength,
            z=ray.z,
        )


@jdc.pytree_dataclass
class DoubleDeflector:
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
class Biprism:
    z: float
    offset: float = 0.0
    rotation: Degrees = 0.0
    deflection: float = 0.0

    def step(
        self,
        ray: Ray,
    ) -> Ray:
        pos_x, pos_y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        deflection = self.deflection
        offset = self.offset
        rot = jnp.deg2rad(self.rotation)

        rays_v = jnp.array([pos_x, pos_y]).T

        biprism_loc_v = jnp.array([offset * jnp.cos(rot), offset * jnp.sin(rot)])

        biprism_v = jnp.array([-jnp.sin(rot), jnp.cos(rot)])
        biprism_v /= jnp.linalg.norm(biprism_v)

        rays_v_centred = rays_v - biprism_loc_v

        dot_product = jnp.dot(rays_v_centred, biprism_v) / jnp.dot(biprism_v, biprism_v)
        projection = jnp.outer(dot_product, biprism_v)

        rejection = rays_v_centred - projection
        rejection = rejection / jnp.linalg.norm(rejection, axis=1, keepdims=True)

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

        return Ray(
            x=pos_x.squeeze(),
            y=pos_y.squeeze(),
            dx=new_dx,
            dy=new_dy,
            _one=ray._one,
            pathlength=pathlength,
            z=ray.z,
        )


# Base class for grid transforms


# @jdc.pytree_dataclass
# class ImageGrid(GridBase):
#     z: float
#     image_pixel_size: Scale_YX
#     image_shape: Shape_YX
#     image_rotation: Degrees
#     image_centre: Coords_XY = (0., 0.)
#     image_array: jnp.ndarray = None  # Added image array variable specific to ImageGrid
#     metres_to_pixels_mat: jnp.ndarray = jdc.field(init=False)
#     pixels_to_metres_mat: jnp.ndarray = jdc.field(init=False)

#     @property
#     def pixel_size(self) -> Scale_YX:
#         return self.image_pixel_size

#     @property
#     def shape(self) -> Shape_YX:
#         return self.image_shape

#     @property
#     def rotation(self) -> Degrees:
#         return self.image_rotation

#     @property
#     def centre(self) -> Coords_XY:
#         return self.image_centre
