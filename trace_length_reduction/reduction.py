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


def Y_k(alpha,beta,k):
    return (beta, (alpha**(-1))*(beta**k))

def X_k(alpha,beta,k):
    return (beta, alpha*(beta**k))


def diagonalise_in_order(B, precision=100):
    """
    Assumes B hyperbolic and diagonalises B such that B = PDP^(-1) and D = diag(l1,l2,l3), l1 > l2 > l3.
    """
    ls, vs = [],[]
    eigenvectors = B.eigenvects()
    assert len(eigenvectors) == 3, "B does not have 3 distinct eigenvectors"
    for eigenvector in B.eigenvects():
        ls.append(eigenvector[0])
        vs.append(eigenvector[2][0])
    ls_precision = [N(ls[i],precision).as_real_imag()[0] for i in range(3)]
    argsort = np.argsort(ls_precision)[::-1] # sorts eigenvalues such that first is largest
    ls = np.array(ls)[argsort]
    vs = np.array(vs)[argsort]
    D = sp.Matrix([[ls[0],0,0],[0,ls[1],0],[0,0,ls[2]]])
    P = sp.Matrix(np.hstack([*vs]))
    P,D = N(P,precision).as_real_imag()[0], N(D,precision).as_real_imag()[0] # converts to real matrix
    return (P,D)


def get_k0_k1(A,B, s, move):

    P,D = diagonalise_in_order(B)

    if move == Y_k:
        A = A.inv()

    A_tilde = P.inv()*A*P

    a,b,c = A_tilde[0,0], A_tilde[1,1], A_tilde[2,2]
    
    l1,l2,l3 = D[0,0], D[1,1], D[2,2]

    if b!=0:
        k1 = int(sp.ceiling(sp.Max(0,
            sp.log(a/2/(2*sp.Abs(b)))/sp.log(l2/l1),
            sp.log(a/2/(2*c))/sp.log(l3/l1),
            sp.log(s/(a/2))/sp.log(l1))))
        k0 = -int(sp.ceiling(sp.Max(0,
            sp.log(c/2/(2*sp.Abs(b)))/sp.log(l3/l2),
            sp.log(c/2/(2*a))/sp.log(l3/l1),
            sp.log(s/(c/2))/sp.log(1/l3))))
    
    else:
        k1 = int(sp.ceiling(sp.Max(0,
            sp.log(a/2/c)/sp.log(l3/l1),
            sp.log(s/(a/2))/sp.log(l1))
        ))
        k0 = -int(sp.ceiling(sp.Max(0,
            sp.log(c/2/a)/sp.log(l3/l1),
            sp.log(s/(c/2))/sp.log(1/l3))
        ))

    return [k0,k1]



def get_bounds(alpha,beta,objective,move, delta=0):
    """
    Constructs bounds [k0,k1] such that the complement [k0,k1] necessarily guarantees the values are negative. If no values exist, it returns an empty array. Assumes the value of trace is strictly concave up and has a global minimiser.
    """
    objective_function, objective_label = objective


    if objective_label == 'trace':
        s = sp.trace(beta) + delta
    else:
        s = trmax(objective_function(beta) + delta) # lmin^(-1)(l(beta))


    k0,k1 = get_k0_k1(alpha,beta,s,move)

    def f(k):
        return s - sp.trace(move(alpha,beta,k)[1])
    
    return [k0, k1]

def main_algorithm(alpha, beta, objective, visited_generators=[],expression=(sp.Symbol('a',commutative=False),sp.Symbol('b',commutative=False)), verbose=False):
    assert isinstance(alpha, sp.Matrix), "Error: alpha is not a sp.Matrix"
    assert isinstance(beta, sp.Matrix), "Error: beta is not a sp.Matrix"


    objective_function, objective_label = objective

    if objective_function(alpha) < objective_function(beta):
        alpha,beta = beta,alpha

        
    def objective_difference(A,B):
        return objective_function(A) - objective_function(B)

    def f(k):
        return objective_difference(*X_k(alpha,beta,k))

    def g(k):
        return objective_difference(*Y_k(alpha,beta,k))

    visited_generators.append((alpha,beta))
    
    candidate_moves = [] # elements are tuples with powers, then X_k or Y_k
    
    bounds = get_bounds(alpha,beta, objective, X_k)

    if bounds:
        [k0_algorithm,k1_algorithm] = bounds
        k_vals = np.linspace(k0_algorithm, k1_algorithm, k1_algorithm-k0_algorithm+1, dtype=np.int64)

        positive_k_vals = []
        for k in k_vals:
            if f(k)>0:
                positive_k_vals.append(k)

        if len(positive_k_vals):
            for k in positive_k_vals:
                candidate_moves.append((k, X_k, f(k)))
    
    bounds = get_bounds(alpha,beta, objective, Y_k)

    if bounds:
        [k0_algorithm,k1_algorithm] = bounds
        k_vals = np.linspace(k0_algorithm, k1_algorithm, k1_algorithm-k0_algorithm+1, dtype=np.int64)
        
        positive_k_vals = []
        for k in k_vals:
            if g(k)>0:
                positive_k_vals.append(k)

        if len(positive_k_vals):
            for k in positive_k_vals:
                candidate_moves.append((k, Y_k, g(k)))

    if verbose:   
        print(candidate_moves, 'candidate moves', [float(candidate_moves[i][-1]) for i in range(len(candidate_moves))])

    
    if candidate_moves:
        values = [candidate_moves[i][2] for i in range(len(candidate_moves))]
        best_move = candidate_moves[np.argmax(values)]
        k = best_move[0]
        move_function = best_move[1]

        expression = move_function(expression[0],expression[1], k)

        if verbose:
            print("algorithm move applied:", f"{str(move_function).rsplit(" ")[1]} {k}", expression)


        alpha_prime, beta_prime, visited_generators,expression = main_algorithm(*move_function(alpha,beta, k), objective,visited_generators, expression)

    else:
        alpha_prime,beta_prime = alpha,beta
        if verbose:
            print('no candidate moves')


    #### post processing


    def f(k):
        return objective_difference(*X_k(alpha_prime, beta_prime,k)) + objective_difference(alpha_prime,beta_prime)
    
    def g(k):
        return objective_difference(*Y_k(alpha_prime, beta_prime,k)) + objective_difference(alpha_prime,beta_prime)


    candidate_moves = [] # elements are tuples with powers, then X_k or Y_k
    
    bounds = get_bounds(alpha_prime,beta_prime,objective, X_k, delta=objective_difference(alpha_prime,beta_prime))

    if bounds:
        [k0_algorithm,k1_algorithm] = bounds
        k_vals =  np.linspace(k0_algorithm, k1_algorithm, k1_algorithm-k0_algorithm+1, dtype=np.int64)
        if len(k_vals):
            for k in k_vals:
                candidate_moves.append((k, X_k, f(k)))

    bounds = get_bounds(alpha_prime,beta_prime,objective,Y_k,delta=objective_difference(alpha_prime,beta_prime))
    if bounds:
        [k0_algorithm,k1_algorithm] = bounds
        k_vals = np.linspace(k0_algorithm, k1_algorithm, k1_algorithm-k0_algorithm+1, dtype=np.int64)
        if len(k_vals):
            for k in k_vals:
                candidate_moves.append((k, Y_k, g(k)))

    if verbose:        
        print(candidate_moves, 'post processing candidate moves', [float(candidate_moves[i][-1]) for i in range(len(candidate_moves))])

    
    if candidate_moves:
        values = [candidate_moves[i][2] for i in range(len(candidate_moves))]
        best_move = candidate_moves[np.argmax(values)]
        k = best_move[0]
        move_function = best_move[1]

        expression = X_k(*move_function(expression[0],expression[1], k),0)

        if verbose:
            print("post processing algorithm move applied:", f"{str(move_function).rsplit(" ")[1]} {k}", expression)

        alpha_prime, beta_prime = X_k(*move_function(alpha_prime,beta_prime, k),0)

    else:
        if verbose:
            print('no candidate post processing moves')

    return alpha_prime, beta_prime, visited_generators, expression


class XCoords():
    def __init__(self, coords, cube_roots=[]):
        assert isinstance(coords, list), f"Error: {coords} must be a list. It is {coords.__class__}."
        assert isinstance(cube_roots, list), f"Error: {cube_roots} must be a list. It is {cube_roots.__class__}."

        if not len(cube_roots):
            cube_roots = [sp.Pow(coord, 1/sp.Number(3)) for coord in coords]
        
        assert len(coords) == 8, f"Error: {coords} must contain 8 coordinates. It has {len(coords)}."
        assert len(cube_roots) == 8, f"Error: {cube_roots} must contain 8 coordinates. It has {len(cube_roots)}."
        assert np.all([isinstance(coords[i], sp.Expr) for i in range(len(coords))]), f"Error: {coords} must be instances of sympy numbers or sympy expressions. They are {[coords[i].__class__ for i in range(len(coords))]}"
        assert np.all([isinstance(cube_roots[i], sp.Expr) for i in range(len(cube_roots))]), f"Error: {cube_roots} must be instances of sympy numbers or powers. They are {[cube_roots[i].__class__ for i in range(len(cube_roots))]}"
        assert np.all([sp.Pow(cube_roots[i],3) for i in range(len(cube_roots))] == coords), "Error: cube roots are not cube roots of coords."

        self.coords = coords
        self.cube_roots = cube_roots

    def get_coords(self):
        return (self.coords, self.cube_roots)

class ReductionResults():
    # TO DO: add metrics, expressions and post processing results in a self-contained report, with options to render as LaTeX
    def __init__(self):
        pass

class Objective():
    def __init__(self, objective_label):
        assert objective_label == 'trace' or objective_label == 'length', f"Error: objective_label {objective_label} is not 'trace' or 'length'."
        self.objective_label = objective_label
        if self.objective_label == 'trace':
            self.objective_function = sp.trace 
        else:
            self.objective_function = calculate_geodesic_length

    def get_objective(self):
        return (self.objective_function, self.objective_label)
        
class TraceLengthReductionInterface():
    def __init__(self, x):
        assert isinstance(x, XCoords), "Error: x must be an instance of XCoords."
        coords, cube_roots = x.get_coords()
        self.generators = compute_translation_matrix_torus(*coords, *cube_roots)
    
    def trace_reduction(self, verbose=False):
        alpha_returned, beta_returned, visited_generators_trace, expression = main_algorithm(*self.generators, Objective('trace').get_objective(), verbose=verbose)
        # print(expression)
        # print(visited_generators_trace)

    def length_reduction(self, verbose=False):
        alpha_returned, beta_returned, visited_generators_length, expression = main_algorithm(*self.generators, Objective('length').get_objective(), verbose=verbose)


# random_integers = [1,1]*2 +[15, 10]*6 # special end longer
# random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
# random_rationals[2]=sp.Number(1)/random_rationals[3]
# random_rationals[4]=sp.Number(1)/random_rationals[5]
# random_rationals[6]=sp.Number(1)/random_rationals[7]
# random_rationals[1] = sp.Number(1)/(sp.Number(2)*random_rationals[0])

# # # TraceLengthReductionInterface(XCoords(random_rationals)).trace_reduction()

# TraceLengthReductionInterface(XCoords([sp.Number(1)]*8)).trace_reduction()