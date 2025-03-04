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

def calculate_canonical_generators(alpha,beta, length_function=calculate_geodesic_length, trace_function=sp.trace):
    """
    Places free basis (alpha,beta) into canonical generators such that
    length(alpha') >= length(beta')
    tr(alpha') >= tr(beta')
    where alpha',beta' belong to the set {alpha,alpha^(-1), beta, beta^(-1)} and remain a free basis
    """
    if length_function(alpha) < length_function(beta):
        alpha,beta = beta,alpha

    if trace_function(beta) > trace_function(beta.inv()):
        beta = beta.inv()
    
    if trace_function(alpha) > trace_function(alpha.inv()):
        alpha = alpha.inv()

    if trace_function(alpha) >= trace_function(beta):
        return (alpha,beta)
    else:
        return (alpha.inv(),beta)

def trmax(x):
    """
    Returns trmax function
    """
    return (sp.exp(x)+2)*sp.exp(-x/sp.Number(3))


def Y_k(alpha,beta,k):
    return (beta, (alpha**(-1))*(beta**k))

def X_k(alpha,beta,k):
    return (beta, alpha*(beta**k))


def commutator(alpha,beta):
    return alpha*beta*(alpha.inv())*(beta.inv())

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

def main_algorithm(alpha, beta, objective, visited_generators=[],expressions=[(sp.Symbol('A',commutative=False),sp.Symbol('B',commutative=False))], moves_applied=[], verbose=False):
    assert isinstance(alpha, sp.Matrix), "Error: alpha is not a sp.Matrix"
    assert isinstance(beta, sp.Matrix), "Error: beta is not a sp.Matrix"


    objective_function, objective_label = objective

    if objective_function(alpha) < objective_function(beta): # only called at the start
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

        expressions.append(move_function(expressions[-1][0],expressions[-1][1], k))
        moves_applied.append((f"{str(move_function).rsplit(" ")[1]}".replace("k", "{" + str(k) + "}"), best_move[-1], False)) # puts string representation of move applied and the objective value difference, i.e. (Y_{-1}, 75409/768)

        if verbose:
            print("algorithm move applied:", f"{str(move_function).rsplit(" ")[1]} {k}", expressions[-1])


        alpha_prime, beta_prime, visited_generators,expressions, moves_applied = main_algorithm(*move_function(alpha,beta, k), objective,visited_generators, expressions, moves_applied)

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
        positive_k_vals = []
        for k in k_vals:
            if f(k)>0:
                positive_k_vals.append(k)

        if len(k_vals):
            for k in positive_k_vals:
                candidate_moves.append((k, X_k, f(k)))

    bounds = get_bounds(alpha_prime,beta_prime,objective,Y_k,delta=objective_difference(alpha_prime,beta_prime))
    if bounds:
        [k0_algorithm,k1_algorithm] = bounds
        k_vals = np.linspace(k0_algorithm, k1_algorithm, k1_algorithm-k0_algorithm+1, dtype=np.int64)
        positive_k_vals = []
        for k in k_vals:
            if g(k)>0:
                positive_k_vals.append(k)
        
        if len(k_vals):
            for k in positive_k_vals:
                candidate_moves.append((k, Y_k, g(k)))

    if verbose:        
        print(candidate_moves, 'post processing candidate moves', [float(candidate_moves[i][-1]) for i in range(len(candidate_moves))])

    
    if candidate_moves:
        values = [candidate_moves[i][2] for i in range(len(candidate_moves))]
        best_move = candidate_moves[np.argmax(values)]
        k = best_move[0]
        move_function = best_move[1]

        expressions.append(X_k(*move_function(expressions[-1][0],expressions[-1][1], k),0))
        moves_applied.append(("X_0 \\circ "+f"{str(move_function).rsplit(" ")[1]}".replace("k", "{" + str(k) + "}"), best_move[-1], True)) # move string, objective value, is post processing


        if verbose:
            print("post processing algorithm move applied:", f"{str(move_function).rsplit(" ")[1]} {k}", expressions[-1])

        alpha_prime, beta_prime = X_k(*move_function(alpha_prime,beta_prime, k),0)

    else:
        if verbose:
            print('no candidate post processing moves')

    return alpha_prime, beta_prime, visited_generators, expressions, moves_applied


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
    def __init__(self, xcoords, objective, returned_expression, returned_generators, initial_generators, pre_canonical_generators, initial_expression=(sp.Symbol('A',commutative=False),sp.Symbol('B',commutative=False))):
        assert isinstance(xcoords, XCoords), "Error: xcoords must be an instance of XCoords."
        assert isinstance(objective, Objective), "Error: objective must be an instance of Objective."
        assert isinstance(returned_expression, tuple) and len(returned_expression) == 2, "Error: Returned expression must be a tuple with two elements."
        assert isinstance(returned_expression[0], sp.Expr) and isinstance(returned_expression[1], sp.Expr), "Error: Returned expression must contain sympy expressions."
        assert isinstance(returned_generators, tuple) and len(returned_generators) == 2, "Error: Returned generators must be a tuple with two elements."
        assert isinstance(returned_generators[0], sp.Matrix) and isinstance(returned_generators[1], sp.Matrix), "Error: Returned generators must contain sympy Matrices."

        self._report = {
                        "xcoords": tuple(xcoords.get_coords()[0]),
                        "xcoords_cube_roots": tuple(xcoords.get_coords()[1]),
                        "objective": objective.get_objective()[1],
                        "moves_applied": [],
                        "expressions": [],
                        "move_readable_strings": [],
                        "visited_generators": [],
                        "pre_canonical_generators": pre_canonical_generators,
                        "initial_expression": initial_expression,
                        "initial_generators": initial_generators,
                        "returned_expression": returned_expression,
                        "returned_generators": returned_generators,
                    }
        
        self._reduction_type_to_latex_hash = {"trace":"\\text{tr}", "length": "\\ell"}
        self._reduction_type_to_str_hash = {"trace":"tr", "length": "l"}

    def get_objective(self):
        return self._report["objective"]

    def _move_to_string_rep(self, move_string,objective_value, is_post_processing, expression, latex=True):
        subscript = self._reduction_type_to_latex_hash[self.get_objective()] if latex else self._reduction_type_to_str_hash[self.get_objective()]
        move_string_objective_value_str = ""

        if not is_post_processing:
            move_string_objective_value_str = f"||{move_string}{str(expression)}||" + "_{" + str(subscript) + "} = " + str(objective_value)
        else:
            move_string_objective_value_str = f"||{move_string.rsplit("circ ")[1]}{str(expression)}||" + "_{" + str(subscript) + "} + " +  f"||{str(expression)}||" + "_{" + str(subscript) + "} = " + str(objective_value)
        return f"Move: ${move_string}$, Objective: ${move_string_objective_value_str}$"

    def update_moves_applied(self, moves_applied, expressions, latex=True):
        
        assert isinstance(moves_applied, list), "Error: moves_applied must be a list."
        for i in range(len(moves_applied)):
            assert isinstance(moves_applied[i], tuple) and len(moves_applied[i]) == 3, "Error: moves_applied must contain tuples with two elements."
            assert isinstance(moves_applied[i][0], str) and (isinstance(moves_applied[i][1], sp.Expr) or isinstance(moves_applied[i][1], float)) and isinstance(moves_applied[i][2], bool), "Error: each tuple in moves_applied must be a string followed by a sympy expression or float and a boolean."
        
        assert isinstance(expressions, list), "Error: expressions must be a list."
        for i in range(len(expressions)):
            assert isinstance(expressions[i], tuple) and len(expressions[i]) == 2, "Error: expressions must contain tuples of length 2."
            assert isinstance(expressions[i][0], sp.Expr) and isinstance(expressions[i][1], sp.Expr), "Error: expressions must contain tuples of sympy expressions."

        move_readable_strings = []

        for i, (move_string, value, is_post_processing) in enumerate(moves_applied):
            move_string_rep = self._move_to_string_rep(move_string, round(float(value),3), is_post_processing, expressions[i], latex)
            move_readable_strings.append(move_string_rep)

        self._report["move_readable_strings"] = move_readable_strings
        self._report["expressions"] = expressions
        self._report["moves_applied"] = moves_applied
        
        return self._report.copy()

    def update_visited_generators(self, visited_generators):
        assert isinstance(visited_generators, list), "Error: visited_generators must be a list."
        for i in range(len(visited_generators)):
            assert isinstance(visited_generators[i], tuple) and len(visited_generators[i]) == 2, "Error: moves_applied must contain tuples with two elements."
            assert isinstance(visited_generators[i][0], sp.Matrix) and isinstance(visited_generators[i][1], sp.Matrix), "Error: each tuple in visited_generators must be a sympy matrix."
        
        self._report["visited_generators"] = visited_generators

        return self._report.copy()
    
    def get_report(self):
        return self._report.copy()
    

    def rational_to_latex_fraction(self, rational):
        string_split = str(rational).rsplit("/")
        if len(string_split)>1:
            return "\\frac" + "{" + string_split[0] + "}{" + string_split[1] + "}"
        else:
            return str(rational)
    
    def get_metrics(self, latex=False):

        alpha_returned, beta_returned = self._report["returned_generators"]

        if not latex:
            return {'X-coordinates': str(self._report["xcoords"]),
                    "(A',B')": str(self._report["returned_expression"]),
                        "tr(B')": np.float64(sp.trace(beta_returned).evalf()),
                        "tr(A')": np.float64(sp.trace(alpha_returned).evalf()),
                        "tr(A'B')": np.float64(sp.trace(alpha_returned*beta_returned).evalf()), 
                        "tr(A'(B')^(-1))":np.float64(sp.trace(alpha_returned*(beta_returned.inv())).evalf()),
                        "tr([A',B'])": np.float64(sp.trace(commutator(alpha_returned,beta_returned)).evalf()),
                        "length(B')":calculate_geodesic_length(beta_returned),
                        "length(A')":calculate_geodesic_length(alpha_returned),
                        "length(A'B')": calculate_geodesic_length(alpha_returned*beta_returned),
                        "length(A'(B')^(-1))":calculate_geodesic_length(alpha_returned*(beta_returned.inv())),
                        "length([A',B'])":calculate_geodesic_length(commutator(alpha_returned,beta_returned))}
        
        else:
            coords = str(self._report["xcoords"])
            return {'$\mathcal{X}$-coordinates': "$" + str(tuple([self.rational_to_latex_fraction(x) for x in coords])).replace("'","").replace("\\\\","\\").replace("(","\\left(").replace(")","\\right)") + "$",
                    "$(A',B')$": "$" + str(self._report["returned_expression"]).replace("**","^").replace("*","") + "$", 
                        "$\\text{tr}(B')$": np.float64(sp.trace(beta_returned).evalf()),
                        "$\\text{tr}(A')$": np.float64(sp.trace(alpha_returned).evalf()),
                        "$\\text{tr}(A'B')$": np.float64(sp.trace(alpha_returned*beta_returned).evalf()),
                        "$\\text{tr}(A'(B')^{-1})$":np.float64(sp.trace(alpha_returned*(beta_returned.inv())).evalf()),
                        "$\\text{tr}([A',B'])$": np.float64(sp.trace(commutator(alpha_returned,beta_returned)).evalf()),
                        "$\\ell(B')$":calculate_geodesic_length(beta_returned),
                        "$\\ell(A')$":calculate_geodesic_length(alpha_returned),
                        "$\\ell(A'B')$": calculate_geodesic_length(alpha_returned*beta_returned),
                        "$\\ell(A'(B')^{-1})$":calculate_geodesic_length(alpha_returned*(beta_returned.inv())),
                        "$\\ell([A',B'])$":calculate_geodesic_length(commutator(alpha_returned,beta_returned))}
            


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
        self.x = x
        coords, cube_roots = x.get_coords()
        self.generators = compute_translation_matrix_torus(*coords, *cube_roots)
        self.canonical_generators = calculate_canonical_generators(*self.generators)

    
    
    
    def trace_reduction(self, verbose=False) -> ReductionResults:
        objective = Objective('trace')
        alpha_returned, beta_returned, visited_generators_trace, expressions, moves_applied = main_algorithm(*self.canonical_generators, objective.get_objective(), verbose=verbose)
        
        reduction_results = ReductionResults(xcoords=self.x,
                                            objective=objective,
                                            returned_expression=expressions[-1],
                                            returned_generators=(alpha_returned, beta_returned),
                                            initial_generators=tuple(self.canonical_generators),
                                            pre_canonical_generators=tuple(self.generators))
        reduction_results.update_moves_applied(moves_applied, expressions)
        reduction_results.update_visited_generators(visited_generators_trace)

        # print(reduction_results.get_report())
        # print(expressions)
        # print(reduction_results.get_metrics())
        # print(moves_applied)
        
        # print(visited_generators_trace)

        return reduction_results

    def length_reduction(self, verbose=False) -> ReductionResults:
        objective = Objective('length')
        alpha_returned, beta_returned, visited_generators_length, expressions, moves_applied = main_algorithm(*self.canonical_generators, objective.get_objective(), verbose=verbose)
        reduction_results = ReductionResults(xcoords=self.x,
                                            objective=objective,
                                            returned_expression=expressions[-1],
                                            returned_generators=(alpha_returned, beta_returned),
                                            initial_generators=tuple(self.canonical_generators),
                                            pre_canonical_generators=tuple(self.generators))
        reduction_results.update_moves_applied(moves_applied, expressions)
        reduction_results.update_visited_generators(visited_generators_length)

        # print(expressions)
        # print(moves_applied)

        return reduction_results

