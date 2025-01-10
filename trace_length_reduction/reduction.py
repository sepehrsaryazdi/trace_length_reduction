import numpy as np
import sympy as sp
from sympy import N


def compute_translation_matrix_torus(A,B,a_minus,a_plus,b_minus,b_plus,e_minus,e_plus, A_cube_root, B_cube_root, a_minus_cube_root, a_plus_cube_root, b_minus_cube_root, b_plus_cube_root, e_minus_cube_root, e_plus_cube_root):
    alpha1 = (triangle_matrix(A,A_cube_root).inv())*edge_matrix(e_minus, e_plus, e_minus_cube_root, e_plus_cube_root)*triangle_matrix(B, B_cube_root)*edge_matrix(a_plus, a_minus, a_plus_cube_root, a_minus_cube_root)
    alpha2 = triangle_matrix(A,A_cube_root)*edge_matrix(b_minus, b_plus, b_minus_cube_root,b_plus_cube_root)*triangle_matrix(B,B_cube_root)*edge_matrix(e_plus, e_minus, e_plus_cube_root, e_minus_cube_root)*triangle_matrix(A,A_cube_root)
    return [alpha1,alpha2]

def edge_matrix(q_plus,q_minus, q_plus_cube_root, q_minus_cube_root):
    coefficient = q_plus_cube_root/q_minus_cube_root
    matrix = sp.Matrix([[sp.Number(0),sp.Number(0),q_minus],[sp.Number(0),-sp.Number(1),sp.Number(0)],[1/q_plus, sp.Number(0), sp.Number(0)]])
    return coefficient*matrix

def triangle_matrix(t, t_cube_root):
    coefficient = sp.Number(1)/t_cube_root
    matrix = sp.Matrix([[sp.Number(0),sp.Number(0),sp.Number(1)],[sp.Number(0),-sp.Number(1),-sp.Number(1)],[t,t+sp.Number(1),sp.Number(1)]])
    return matrix*coefficient

def compute_t(a_minus, b_minus, e_minus, a_plus, b_plus, e_plus):
    return a_minus*b_minus*e_minus/(a_plus*b_plus*e_plus)

def compute_q_plus(A, d_minus, B, a_minus):
    return A*d_minus/(B*a_minus)

def a_to_x_coordinate_torus(A,B,a_minus,a_plus,b_minus,b_plus,e_minus,e_plus):
    qe_plus = compute_q_plus(A,b_minus, B, a_minus)
    qe_minus = compute_q_plus(B,b_plus, A, a_plus)
    A_t = compute_t(a_minus, b_minus, e_minus, a_plus, b_plus, e_plus)
    B_t = compute_t(e_plus, a_plus, b_plus, e_minus, a_minus, b_minus)
    qb_plus = compute_q_plus(A, a_minus, B, e_minus)
    qb_minus = compute_q_plus(B, a_plus, A, e_plus)
    qa_plus = compute_q_plus(A,e_minus, B, b_minus)
    qa_minus = compute_q_plus(B,e_plus, A, b_plus)
    return (A_t,B_t,qa_minus, qa_plus, qb_minus, qb_plus, qe_minus, qe_plus)


def calculate_geodesic_length_fast(gamma): # faster implementation of length that may introduce floating point errors
    gamma_eigenvalues =  np.array(sp.Matrix(list(gamma.eigenvals())).T.evalf())
    gamma_eigenvalues = np.sort(np.abs(gamma_eigenvalues)).flatten()[::-1]
    gamma_length = np.log(np.float64(max(gamma_eigenvalues)/min(gamma_eigenvalues)))
    return gamma_length

def calculate_geodesic_length_high_precision(gamma,precision=100): # slower implementation that maintains high precision and only rounds at the end
    gamma_eigenvalues = list(gamma.eigenvals())
    eigenvalues_to_precision = [N(gamma_eigenvalues[i],precision).as_real_imag()[0] for i in range(len(gamma_eigenvalues))] # note that the real part is taken, the imaginary part is theoretically zero and negligible for a given rounded number.
    return np.float64(sp.log(sp.Max(*eigenvalues_to_precision)/sp.Min(*eigenvalues_to_precision)))

def calculate_geodesic_length(gamma):
    return calculate_geodesic_length_high_precision(gamma)


def calculate_lengths(alpha,beta):
    return calculate_geodesic_length(alpha), calculate_geodesic_length(beta)

def trmax(x):
    """
    Returns trmax function
    """
    return (sp.exp(x)+2)*sp.exp(-x/sp.Number(3))



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


class ReductionResults():
    def __init__(self):
        pass

class TraceLengthObj():
    def __init__(self, x):
        assert isinstance(x, XCoords), "Error: x must be an instance of XCoords."
        coords, cube_roots = x.get_coords()
        self.generators = compute_translation_matrix_torus(*coords, *cube_roots)

TraceLengthObj(XCoords([sp.Number(1)]*8))