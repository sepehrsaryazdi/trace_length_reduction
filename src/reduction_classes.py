import numpy as np
import sympy as sp

class XCoords():
    def __init__(self, coords, cube_roots=[]):
        assert isinstance(coords, list), f"Error: {coords} must be a list. It is {coords.__class__}."
        assert isinstance(cube_roots, list), f"Error: {cube_roots} must be a list. It is {cube_roots.__class__}."

        if not len(cube_roots):
            cube_roots = [sp.Pow(coord, 1/sp.Number(3)) for coord in coords]
        
        assert len(coords) == 8, f"Error: {coords} must contain 8 coordinates. It has {len(coords)}."
        assert len(cube_roots) == 8, f"Error: {cube_roots} must contain 8 coordinates. It has {len(cube_roots)}."
        assert np.all([(isinstance(coords[i], sp.Number) or isinstance(coords[i], sp.core.power.Pow)) for i in range(len(coords))]), f"Error: {coords} must be instances of sympy numbers or powers. They are {[coords[i].__class__ for i in range(len(coords))]}"
        assert np.all([(isinstance(cube_roots[i], sp.Number) or isinstance(cube_roots[i], sp.core.power.Pow)) for i in range(len(cube_roots))]), f"Error: {cube_roots} must be instances of sympy numbers or powers. They are {[cube_roots[i].__class__ for i in range(len(cube_roots))]}"
        assert np.all([sp.Pow(cube_roots[i],3) for i in range(len(cube_roots))] == coords), "Error: cube roots are not cube roots of coords."

        self.coords = coords
        self.cube_roots = cube_roots

    def get_coords(self):
        return (self.coords, self.cube_roots)


class OncePuncturedTorusGenerators():
    def __init__(self, x):
        assert isinstance(x, XCoords), "Error: x must be an instance of XCoords."

class ReductionResults():
    def __init__(self):
        pass

class LengthTraceObj():
    def __init__(self, x):
        assert isinstance(x, XCoords), "Error: x must be an instance of XCoords."
