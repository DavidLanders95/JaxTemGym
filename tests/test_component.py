import pytest
import jax
import numpy as np
from numpy.testing import assert_allclose

from jaxgym.components import (
    MatrixComponent,
    ScanGrid, Detector, DescanError,
    Identity, FreeSpace, Lens, ThickLens, Shifter, Tilter,
    scan_descan_triplet,
)
from jaxgym.ray import Ray, RayMatrix, unwrap, ray_matrix

jax.config.update('jax_platform_name', 'cpu')


def test_identity():
    c = Identity()
    c.validate()

    sample = Ray(
        x=0., y=0., z=0., dx=0., dy=0.
    )
    sample_2 = Ray(
        x=1., y=2., z=3., dx=4., dy=5.
    )

    ref = c.step(sample)
    ref_2 = c.step(sample_2)
    # Actually identity
    assert ref == sample
    assert ref_2 == sample_2

    assert c.pathlength(sample) == 0


def test_free_space():
    c = FreeSpace(length=3)
    c.validate()

    sample = Ray(
        x=0., y=0., z=0., dx=0., dy=0.
    )
    sample_2 = Ray(
        x=1., y=2., z=3., dx=4., dy=5.
    )

    res = c.step(sample)
    ref = sample.modify(z=3)
    assert_allclose(unwrap(ref), unwrap(res))
    assert_allclose(c.pathlength(sample), c.length)

    res_2 = c.step(sample_2)
    ref_2 = sample_2.modify(
        z=sample_2.z + c.length,
        x=sample_2.x + sample_2.dx*c.length,
        y=sample_2.y + sample_2.dy*c.length,
    )
    assert_allclose(unwrap(ref_2), unwrap(res_2))
    assert_allclose(c.pathlength(sample_2), c.length * np.sqrt((1 + sample_2.dx**2 + sample_2.dy**2)))


def test_lens_infinity():
    # A thin lens with infinite focal length
    # has no effect, i.e. is Identity()
    c = Lens(focal_length=np.inf)
    c.validate()

    sample = Ray(
        x=0., y=0., z=0., dx=0., dy=0.
    )
    sample_2 = Ray(
        x=1., y=2., z=3., dx=4., dy=5.
    )

    ref = c.step(sample)
    ref_2 = c.step(sample_2)
    # Actually identity
    assert ref == sample
    assert ref_2 == sample_2

    assert c.pathlength(sample) == 0
    assert c.pathlength(sample_2) == 0


def test_lens():
    focal_length = 23
    c = Lens(focal_length=focal_length)
    c.validate()

    # A ray through the center of a lens is not modified
    central = Ray(
        x=0., y=0., z=0., dx=0.1, dy=0.3
    )
    central_res = c.step(central)

    assert_allclose(unwrap(central), unwrap(central_res))
    assert_allclose(c.pathlength(central), 0)

    # A beam parallel to the optical axis passes through the focus point
    parallel = Ray(
        x=1., y=2., z=3., dx=0., dy=0.
    )
    parallel_res = c.step(parallel)
    foc = FreeSpace(length=focal_length)
    at_focus = foc.step(parallel_res)

    assert_allclose(at_focus.x, 0)
    assert_allclose(at_focus.y, 0)
    assert_allclose(at_focus.z, parallel.z + focal_length)
    # TODO put a better test here and not just the formula from the code itself
    assert_allclose(c.pathlength(parallel), -(parallel.x**2 + parallel.y**2) / (2 * focal_length))


def test_thick():
    focal_length = 23
    thickness = 13
    c = ThickLens(focal_length=focal_length, thickness=thickness)
    c.validate()

    # A ray through the center of a lens is not modified
    central = Ray(
        x=0., y=0., z=0., dx=0.1, dy=0.3
    )
    central_res = c.step(central)
    central_ref = central.modify(z=central.z + c.thickness)

    assert_allclose(unwrap(central_ref), unwrap(central_res))
    assert_allclose(c.pathlength(central), 0)

    # A beam parallel to the optical axis passes through the focus point
    parallel = Ray(
        x=1., y=2., z=3., dx=0., dy=0.
    )
    parallel_res = c.step(parallel)
    foc = FreeSpace(length=focal_length)
    at_focus = foc.step(parallel_res)

    assert_allclose(at_focus.x, 0)
    assert_allclose(at_focus.y, 0)
    assert_allclose(at_focus.z, parallel.z + focal_length + c.thickness)
    # TODO put a better test here and not just the formula from the code itself
    assert_allclose(c.pathlength(parallel), -(parallel.x**2 + parallel.y**2) / (2 * focal_length))


def test_shifter():
    c = Shifter(offset_x=2, offset_y=3)
    c.validate()

    sample = Ray(
        x=0., y=0., z=0., dx=0., dy=0.
    )
    sample_2 = Ray(
        x=1., y=2., z=3., dx=4., dy=5.
    )

    res = c.step(sample)
    res_2 = c.step(sample_2)
    ref = sample.modify(
        x=sample.x + c.offset_x,
        y=sample.y + c.offset_y,
    )
    ref_2 = sample_2.modify(
        x=sample_2.x + c.offset_x,
        y=sample_2.y + c.offset_y,
    )

    assert_allclose(unwrap(res), unwrap(ref))
    assert_allclose(unwrap(res_2), unwrap(ref_2))

    assert_allclose(c.pathlength(sample), 0)
    # TODO better validation than just the original formula
    assert_allclose(c.pathlength(sample_2), -(c.offset_x * sample_2.x) - (c.offset_y * sample_2.y))


def test_tilter():
    c = Tilter(
        tilt_x=2, tilt_y=3,
    )
    c.validate()

    sample = Ray(
        x=0., y=0., z=0., dx=0., dy=0.
    )
    sample_2 = Ray(
        x=1., y=2., z=3., dx=4., dy=5.
    )

    res = c.step(sample)
    res_2 = c.step(sample_2)
    ref = sample.modify(
        dx=sample.dx + 2,
        dy=sample.dy + 3,
    )
    ref_2 = sample_2.modify(
        dx=sample_2.dx + 2,
        dy=sample_2.dy + 3,
    )

    assert_allclose(unwrap(res), unwrap(ref))
    assert_allclose(unwrap(res_2), unwrap(ref_2))

    assert_allclose(c.pathlength(sample), 0)
    assert_allclose(c.pathlength(sample_2), 0)


def test_descan_triplet_tilt():
    scan, descan, tilt = scan_descan_triplet(
        offset_x=2, offset_y=3,
        descan_error=DescanError(
            dxx=1, dxy=2, dyx=3, dyy=4
        )
    )

    sample = Ray(
        x=0., y=0., z=0., dx=0., dy=0.
    )
    sample_2 = Ray(
        x=1., y=2., z=3., dx=4., dy=5.
    )

    res = tilt.step(descan.step(scan.step(sample)))
    res_2 = tilt.step(descan.step(scan.step(sample_2)))
    ref = sample.modify(
        dx=sample.dx + 2 * 1 + 3 * 2,
        dy=sample.dy + 3 * 4 + 2 * 3,
    )

    ref_2 = sample_2.modify(
        dx=sample_2.dx + 2 * 1 + 3 * 2,
        dy=sample_2.dy + 3 * 4 + 2 * 3,
    )

    assert_allclose(unwrap(res), unwrap(ref))
    assert_allclose(unwrap(res_2), unwrap(ref_2))


def test_descan_triplet_shift():
    scan, descan, tilt = scan_descan_triplet(
        offset_x=2, offset_y=3,
        descan_error=DescanError(
            xx=1, xy=2, yx=3, yy=4
        )
    )

    sample = Ray(
        x=0., y=0., z=0., dx=0., dy=0.
    )
    sample_2 = Ray(
        x=1., y=2., z=3., dx=4., dy=5.
    )

    res = tilt.step(descan.step(scan.step(sample)))
    res_2 = tilt.step(descan.step(scan.step(sample_2)))
    ref = sample.modify(
        x=sample.x + 2 * 1 + 3 * 2,
        y=sample.y + 3 * 4 + 2 * 3,
    )

    ref_2 = sample_2.modify(
        x=sample_2.x + 2 * 1 + 3 * 2,
        y=sample_2.y + 3 * 4 + 2 * 3,
    )

    assert_allclose(unwrap(res), unwrap(ref))
    assert_allclose(unwrap(res_2), unwrap(ref_2))


# Test that the conversion from user-facing component
# to MatrixComponent works and gives equivalent results
@pytest.mark.parametrize(
    'component', [
        Identity(),
        FreeSpace(length=1.3),
        Lens(focal_length=-23),
        Lens(focal_length=42),
        Lens(focal_length=np.inf),
        ThickLens(focal_length=-23, thickness=13),
        ThickLens(focal_length=42, thickness=23),
        ThickLens(focal_length=np.inf, thickness=0),
        Shifter(offset_x=2, offset_y=3),
        Tilter(tilt_x=2, tilt_y=3),
    ]
)
def test_equivalence(component):
    mat = MatrixComponent.from_component(component)
    print(type(component), "\n", mat.matrix)

    sample = Ray(
        x=0., y=0., z=0., dx=0., dy=0.
    )
    sample_2 = Ray(
        x=1., y=2., z=3., dx=4., dy=5.
    )

    ref = component.step(sample)
    ref_2 = component.step(sample_2)

    rays = RayMatrix.from_rays((sample, sample_2))
    res_mat = mat.step(rays)
    ref_mat = RayMatrix.from_rays((ref, ref_2))

    assert_allclose(res_mat.matrix, ref_mat.matrix)


# Test cases for ScanGrid:
@pytest.mark.parametrize(
    "xy, rotation, expected_pixel_coords",
    [
        # No rotation cases
        ((0.0, 0.0), 0.0, (5, 5)),
        ((-0.5, 0.5), 0.0, (0, 0)),
        ((0.5, -0.5), 0.0, (10, 10)),
        ((0.0, 0.5), 0.0, (0, 5)),
        ((-0.5, 0.0), 0.0, (5, 0)),
        # With rotation cases
        ((0.0, 0.0), 90.0, (5, 5)),
        ((-0.5, 0.5), 90.0, (10, 0)),
        ((0.5, -0.5), 90.0, (0, 10)),
        ((0.0, 0.5), 90.0, (5, 0)),
        ((-0.5, 0.0), 90.0, (10, 5)),
    ],
)
def test_scan_grid_metres_to_pixels(xy, rotation, expected_pixel_coords):
    scan_grid = ScanGrid(
        z=0.0,
        scan_rotation=rotation,
        scan_step=(0.1, 0.1),
        scan_shape=(10, 10),
        scan_centre=(0.0, 0.0)
    )
    pixel_coords_y, pixel_coords_x = scan_grid.metres_to_pixels(xy)
    np.testing.assert_allclose(pixel_coords_y, expected_pixel_coords[0], atol=1e-6)
    np.testing.assert_allclose(pixel_coords_x, expected_pixel_coords[1], atol=1e-6)


# Test cases for ScanGrid:
@pytest.mark.parametrize(
    "pixel_coords, rotation, expected_xy",
    [
        # No rotation cases
        ((5, 5), 0.0, (0.0, 0.0)),
        ((0, 0), 0.0, (-0.5, 0.5)),
        ((10, 10), 0.0, (0.5, -0.5)),
        ((0, 5), 0.0, (0.0, 0.5)),
        ((5, 0), 0.0, (-0.5, 0.0)),
        # With rotation cases
        ((5, 5), 90.0, (0.0, 0.0)),
        ((10, 0), 90.0, (-0.5, 0.5)),
        ((0, 10), 90.0, (0.5, -0.5)),
        ((5, 0), 90.0, (0.0, 0.5)),
        ((10, 5), 90.0, (-0.5, 0.0)),
    ],
)
def test_scan_grid_pixels_to_metres(pixel_coords, rotation, expected_xy):
    scan_grid = ScanGrid(
        z=0.0,
        scan_rotation=rotation,
        scan_step=(0.1, 0.1),
        scan_shape=(10, 10),
        scan_centre=(0.0, 0.0)
    )
    metres_coords_x, metres_coords_y = scan_grid.pixels_to_metres(pixel_coords)
    np.testing.assert_allclose(metres_coords_x, expected_xy[0], atol=1e-6)
    np.testing.assert_allclose(metres_coords_y, expected_xy[1], atol=1e-6)


# Test cases for Detector:
@pytest.mark.parametrize(
    "xy, rotation, expected_pixel_coords",
    [
        # No rotation cases
        ((0.0, 0.0), 0.0, (5, 5)),
        ((-0.5, 0.5), 0.0, (0, 0)),
        ((0.5, -0.5), 0.0, (10, 10)),
        ((0.0, 0.5), 0.0, (0, 5)),
        ((-0.5, 0.0), 0.0, (5, 0)),
        # With rotation cases
        ((0.0, 0.0), 90.0, (5, 5)),
        ((-0.5, 0.5), 90.0, (10, 0)),
        ((0.5, -0.5), 90.0, (0, 10)),
        ((0.0, 0.5), 90.0, (5, 0)),
        ((-0.5, 0.0), 90.0, (10, 5)),
    ],
)
def test_detector_metres_to_pixels(xy, rotation, expected_pixel_coords):
    detector = Detector(
        z=0.0,
        det_pixel_size=(0.1, 0.1),
        det_shape=(10, 10),
        det_centre=(0.0, 0.0),
        det_rotation=rotation,
        flip_y=False,
    )
    pixel_coords_y, pixel_coords_x = detector.metres_to_pixels(xy)
    np.testing.assert_allclose(pixel_coords_y, expected_pixel_coords[0], atol=1e-6)
    np.testing.assert_allclose(pixel_coords_x, expected_pixel_coords[1], atol=1e-6)


# Test cases for Detector:
@pytest.mark.parametrize(
    "pixel_coords, rotation, expected_xy",
    [
        # No rotation cases
        ((5, 5), 0.0, (0.0, 0.0)),
        ((0, 0), 0.0, (-0.5, 0.5)),
        ((10, 10), 0.0, (0.5, -0.5)),
        ((0, 5), 0.0, (0.0, 0.5)),
        ((5, 0), 0.0, (-0.5, 0.0)),
        # With rotation cases
        ((5, 5), 90.0, (0.0, 0.0)),
        ((10, 0), 90.0, (-0.5, 0.5)),
        ((0, 10), 90.0, (0.5, -0.5)),
        ((5, 0), 90.0, (0.0, 0.5)),
        ((10, 5), 90.0, (-0.5, 0.0)),
    ],
)
def test_detector_pixels_to_metres(pixel_coords, rotation, expected_xy):
    detector = Detector(
        z=0.0,
        det_rotation=rotation,
        det_pixel_size=(0.1, 0.1),
        det_shape=(10, 10),
        det_centre=(0.0, 0.0),
        flip_y=False,
    )
    metres_coords_x, metres_coords_y = detector.pixels_to_metres(pixel_coords)
    np.testing.assert_allclose(metres_coords_x, expected_xy[0], atol=1e-6)
    np.testing.assert_allclose(metres_coords_y, expected_xy[1], atol=1e-6)

@pytest.mark.parametrize(
    "offset_xy, input_ray_xy, expected_output_xy",
    [
        ((0.0, 0.0), (1.0, 1.0), (1.0, 1.0)),
        ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)),
        ((-1.0, -1.0), (2.0, 2.0), (1.0, 1.0)),
        ((2.5, -2.5), (-3.5, 3.5), (-1.0, 1.0)),
    ],
)
def test_descanner(offset_xy, input_ray_xy, expected_output_xy):

    input_ray = ray_matrix(
        x=input_ray_xy[0],
        y=input_ray_xy[1],
        dx=0.0,
        dy=0.0,
        z=0.0,
        pathlength=1.0,
    )

    descanner = Descanner(
        z=0.0,
        offset_x=offset_xy[0],
        offset_y=offset_xy[1],
        descan_error=[0,0,0,0,0,0,0,0]
    )

    output_ray = descanner.step(input_ray)

    np.testing.assert_allclose(output_ray.x, expected_output_xy[0], atol=1e-6)
    np.testing.assert_allclose(output_ray.y, expected_output_xy[1], atol=1e-6)

    

