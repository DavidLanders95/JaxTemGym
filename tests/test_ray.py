from jaxgym.ray import Ray, RayMatrix


def test_smoke():
    r1 = Ray(
        x=0, y=0,
        dx=0, dy=0,
    )
    r2 = Ray(
        x=0, y=0,
        dx=0.1, dy=0,
    )
    rays = RayMatrix.from_rays((r1, r2))
    revRays = rays.to_rays()
    assert revRays == (r1, r2)


def test_modify():
    r1 = Ray(
        x=0., y=0., z=0.,
        dx=0., dy=0.,
        pathlength=0.
    )
    for key in ('x', 'y', 'z', 'dx', 'dy', 'pathlength'):
        r2 = r1.modify(**{key: 23})
        for key2 in ('x', 'y', 'z', 'dx', 'dy', 'pathlength'):
            if key == key2:
                ref = 23
            else:
                ref = getattr(r1, key2)
            assert getattr(r2, key2) == ref

