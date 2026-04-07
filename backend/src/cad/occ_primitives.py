"""pythonocc primitive surface builders."""


def make_occ_plane(normal, d):
    """Create an OCC Geom_Plane from normal and offset."""
    from OCC.Core.gp import gp_Pnt, gp_Dir
    from OCC.Core.Geom import Geom_Plane
    import numpy as np

    n = np.array(normal)
    point = -d * n
    return Geom_Plane(
        gp_Pnt(float(point[0]), float(point[1]), float(point[2])),
        gp_Dir(float(n[0]), float(n[1]), float(n[2]))
    )


def make_occ_cylinder(axis, center, radius):
    """Create an OCC Geom_CylindricalSurface."""
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3
    from OCC.Core.Geom import Geom_CylindricalSurface

    return Geom_CylindricalSurface(
        gp_Ax3(
            gp_Pnt(float(center[0]), float(center[1]), float(center[2])),
            gp_Dir(float(axis[0]), float(axis[1]), float(axis[2]))
        ),
        float(radius)
    )


def make_occ_sphere(center, radius):
    """Create an OCC Geom_SphericalSurface."""
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3
    from OCC.Core.Geom import Geom_SphericalSurface

    return Geom_SphericalSurface(
        gp_Ax3(
            gp_Pnt(float(center[0]), float(center[1]), float(center[2])),
            gp_Dir(0, 0, 1)
        ),
        float(radius)
    )
