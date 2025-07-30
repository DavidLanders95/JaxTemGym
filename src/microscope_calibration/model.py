from typing import Optional, NamedTuple, Dict, Union
from collections import OrderedDict
import jax_dataclasses as jdc

from jaxgym.coordinates import XYCoordinateSystem, XYVector
from jaxgym.ray import Ray

from jaxgym.components import (
    Component,
    Sampling, FreeSpace, Descanner, Scanner, DescanError
)
from jaxgym.source import Source, PointSource


@jdc.pytree_dataclass
class Parameters4DSTEM:
    overfocus: float  # m
    scan_pixel_pitch: float  # m
    scan_cy: float  # px
    scan_cx: float  # px
    scan_rotation: float  # rad
    camera_length: float  # m
    detector_pixel_pitch: float  # m
    detector_cy: float  # px
    detector_cx: float  # px
    semiconv: float  # rad
    flip_y: bool
    descan_error: DescanError = DescanError()


class ResultSection(NamedTuple):
    component: Union[Component, Source]
    ray: Ray
    sampling: Optional[Dict] = None


Result4DSTEM = OrderedDict[str, ResultSection]


@jdc.pytree_dataclass
class Model4DSTEM:
    params: Parameters4DSTEM

    def make_source_ray(
            self,
            source_dx: float, source_dy: float,
            _one: float = 1.) -> ResultSection:
        source = PointSource()
        ray = source(dy=source_dy, dx=source_dx, _one=_one)
        return ResultSection(component=source, ray=ray)

    def trace(
            self,
            scan_px_y: float, scan_px_x: float,
            ray: Ray) -> Result4DSTEM:
        params = self.params
        components = OrderedDict()
        scan_to_real = XYCoordinateSystem.identity()\
            .shift(XYVector(x=-params.scan_cx, y=-params.scan_cy))\
            .scale(params.scan_pixel_pitch)\
            .rotate(params.scan_rotation)
        real_to_scan = scan_to_real.invert()
        flip_factor = -1. if params.flip_y else 1.
        detector_to_real = XYCoordinateSystem.identity()\
            .flip_y(params.flip_y)\
            .shift(XYVector(x=-params.detector_cx, y=-flip_factor*params.detector_cy))\
            .scale(params.detector_pixel_pitch)
        real_to_detector = detector_to_real.invert()

        scan_deflection = scan_to_real(XYVector(x=scan_px_x, y=scan_px_y, _one=ray._one))

        components['overfocus'] = FreeSpace(params.overfocus)
        components['scanner'] = Scanner(
            scan_pos_x=scan_deflection.x,
            scan_pos_y=scan_deflection.y,
        )
        components['specimen'] = Sampling()
        components['descanner'] = Descanner(
            scan_pos_x=scan_deflection.x,
            scan_pos_y=scan_deflection.y,
            descan_error=params.descan_error,
        )
        components['camera_length'] = FreeSpace(params.camera_length)
        components['detector'] = Sampling()

        result = OrderedDict()

        for (key, component) in components.items():
            ray = component(ray)
            if key == 'specimen':
                scan_px = real_to_scan(XYVector(x=ray.x, y=ray.y, _one=ray._one))
                result[key] = ResultSection(
                    component=component,
                    ray=ray,
                    sampling={'scan_px': scan_px}
                )
            elif key == 'detector':
                detector_px = real_to_detector(XYVector(x=ray.x, y=ray.y, _one=ray._one))
                result[key] = ResultSection(
                    component=component,
                    ray=ray,
                    sampling={'detector_px': detector_px}
                )
            else:
                result[key] = ResultSection(
                    component=component,
                    ray=ray
                )
        return result
