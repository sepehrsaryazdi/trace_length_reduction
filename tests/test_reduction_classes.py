import unittest
from src.reduction_classes import XCoords
import sympy as sp
import numpy as np

class TestReductionClasses(unittest.TestCase):
    def test_xcoords_initialise(self):
        try:
            x = XCoords([1.1]*8)
            assert False, "Error: XCoords did not catch unallowed case."
        except:
            print("XCoords correctly catches unallowed case.")
            return
    
    def test_xcoords_output(self):
        x = XCoords([sp.Number(2)]*8)
        coords, cube_roots = x.get_coords()
        assert len(coords) == 8, f"Error: {coords} must contain 8 coordinates. It has {len(coords)}."
        assert len(cube_roots) == 8, f"Error: {cube_roots} must contain 8 coordinates. It has {len(cube_roots)}."
        assert np.all([(isinstance(coords[i], sp.Number) or isinstance(coords[i], sp.core.power.Pow)) for i in range(len(coords))]), f"Error: {coords} must be instances of sympy numbers or powers. They are {[coords[i].__class__ for i in range(len(coords))]}"
        assert np.all([(isinstance(cube_roots[i], sp.Number) or isinstance(cube_roots[i], sp.core.power.Pow)) for i in range(len(cube_roots))]), f"Error: {cube_roots} must be instances of sympy numbers or powers. They are {[cube_roots[i].__class__ for i in range(len(cube_roots))]}"
        assert np.all([sp.Pow(cube_roots[i],3) for i in range(len(cube_roots))] == coords), "Error: cube roots are not cube roots of coords."
        print("XCoords has correct output.")

        


if __name__ == "__main__":
    unittest.main()
