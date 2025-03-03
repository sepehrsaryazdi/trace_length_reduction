import numpy as np
import sympy as sp
from sympy import N
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import pandas as pd

from examples.generation_functions import give_all_examples


def represent_matrix_as_latex(M):
    return str(M).replace("Matrix","").replace("(", r"\begin{pmatrix} ").replace(")",r" \end{pmatrix}").replace("],",r" \\").replace(",",r" &").replace("[","").replace("]","")
    

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


def calculate_lengths(alpha,beta):
    return calculate_geodesic_length(alpha), calculate_geodesic_length(beta)

def id(alpha, beta,k):
    return alpha,beta


def Y_k(alpha,beta,k):
    return (beta, (alpha**(-1))*(beta**k))

def X_k(alpha,beta,k):
    return (beta, alpha*(beta**k))

def s(alpha,beta,k):
    return (beta, alpha)

def i1(alpha, beta,k):
    return (alpha.inv(), beta)

def i2(alpha,beta,k):
    return (alpha,beta.inv())

def commutator(alpha,beta):
    return alpha*beta*(alpha.inv())*(beta.inv())

def matrix_inner_product(A,B):
    return sp.trace(A*B)

def embed_into_sl3r(g):
    A = 1/sp.sqrt(2) * sp.Matrix([[1,0],[0,-1]])
    B = 1/sp.sqrt(2) * sp.Matrix([[0,1],[1,0]])
    C = 1/sp.sqrt(2) * sp.Matrix([[0,1],[-1,0]])
    g_inv = g.inv()
    return sp.Matrix([[1,0,0],[0,1,0],[0,0,-1]]) * sp.Matrix([[matrix_inner_product(A,g*A*g_inv), matrix_inner_product(A,g*B*g_inv), matrix_inner_product(A,g*C*g_inv)],
                                                              [matrix_inner_product(B,g*A*g_inv),matrix_inner_product(B,g*B*g_inv),matrix_inner_product(B,g*C*g_inv)],
                                                              [matrix_inner_product(C,g*A*g_inv),matrix_inner_product(C,g*B*g_inv),matrix_inner_product(C,g*C*g_inv)]])




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


def trmax(x):
    """
    Returns trmax function
    """
    return (sp.exp(x)+2)*sp.exp(-x/sp.Number(3))


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
   
    # print(np.isclose(np.array(P*D*(P.inv()),dtype=np.float64), np.array(N(B,precision),dtype=np.float64)))

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

def main_algorithm(alpha, beta, objective, visited_generators=[],expression=(sp.Symbol('a',commutative=False),sp.Symbol('b',commutative=False)), verbose=True):
    assert isinstance(alpha, sp.Matrix), "Error: alpha is not a sp.Matrix"
    assert isinstance(beta, sp.Matrix), "Error: beta is not a sp.Matrix"


    objective_function, objective_label = objective

    if objective_function(alpha) < objective_function(beta):
        print('swapped')
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
        positive_k_vals = []
        for k in k_vals:
            if f(k)>0:
                positive_k_vals.append(k)

        if len(positive_k_vals):
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

        if len(positive_k_vals):
            for k in positive_k_vals:
                candidate_moves.append((k, Y_k, g(k)))

    if verbose:        
        print(candidate_moves, 'post processing candidate moves', [float(candidate_moves[i][-1]) for i in range(len(candidate_moves))])

    
    if candidate_moves:
        values = [candidate_moves[i][2] for i in range(len(candidate_moves))]
        best_move = candidate_moves[np.argmax(values)]
        k = best_move[0]
        move_function = best_move[1]

        print('expression before', expression)
        expression = move_function(expression[0],expression[1], k)

        if verbose:
            print("post processing algorithm move applied:", f"{str(move_function).rsplit(" ")[1]} {k}", expression)

        alpha_prime, beta_prime = X_k(*move_function(alpha_prime,beta_prime, k),0)

    else:
        if verbose:
            print('no candidate post processing moves')

    return alpha_prime, beta_prime, visited_generators, expression



def apply_random_moves(alpha,beta, display=False):
    random_moves = np.vstack([[np.random.randint(-2,2) for i in range(3)],[np.random.randint(0,2) for i in range(3)]]).T
    alpha_mixed, beta_mixed = alpha,beta
    allowed_moves = [id,X_k,Y_k]
    allowed_move_strings = ["id", "X_k", "Y_k", "s", "i1", "i2"]
    for i in range(len(random_moves)):
        power, move_index = random_moves[i]
        alpha_mixed, beta_mixed = allowed_moves[move_index](alpha_mixed,beta_mixed, power)
        if display:
            print("move applied:", allowed_move_strings[move_index], power)
        
    return (alpha_mixed,beta_mixed)




def plot_eigenvalues_and_traces(alpha, beta, visited_generators_trace, visited_generators_length, save_file_name=None,num_distinct_random_group_elements=100, blocking=True):
    """
    Plots the eigenvalues of elements in the group < alpha, beta > and displays them on the length and trace plot, along with the paths taken by the length and trace algorithmsa
    """
    warnings.filterwarnings('ignore')

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

    alpha_original = alpha
    beta_original = beta
    # alpha_returned, beta_returned = apply_random_moves(alpha,beta)

    current_known_fundamental_generators = [(alpha, beta)] + [visited_generators_trace[i] for i in range(len(visited_generators_trace))] + [visited_generators_length[i] for i in range(len(visited_generators_length))] 

    for i in range(1):
        next_generator_to_apply_action_index = np.random.choice([i for i in range(len(current_known_fundamental_generators))])
        next_generator_to_apply_action = current_known_fundamental_generators[int(next_generator_to_apply_action_index)]
        next_generator = apply_random_moves(*next_generator_to_apply_action)

        current_known_fundamental_generators.append(next_generator)


    current_known_fundamental_group_elements = [alpha*beta*(alpha**(-1))*(beta**(-1))] + [alpha,beta] + [visited_generators_trace[i][0] for i in range(len(visited_generators_trace))] + [visited_generators_trace[i][1] for i in range(len(visited_generators_trace))] +  [visited_generators_length[i][0] for i in range(len(visited_generators_length))] + [visited_generators_length[i][1] for i in range(len(visited_generators_length))]
    
    while len(current_known_fundamental_group_elements ) < num_distinct_random_group_elements:
        use_beta = int(np.round(np.random.random_sample())) # if 0, use alpha, otherwise use beta
        power = (-1)**int(np.round(np.random.random_sample())) # if 0, use +1 power. otherwise use -1 power
        action = beta if use_beta else alpha
        action = action.inv() if power < 0 else action
        next_element_to_apply_action_index = np.random.choice([i for i in range(len(current_known_fundamental_group_elements))])
        next_element_to_apply_action = current_known_fundamental_group_elements[int(next_element_to_apply_action_index)]
        new_element = action * next_element_to_apply_action
        if new_element not in current_known_fundamental_group_elements:
            current_known_fundamental_group_elements.append(new_element)
            
    unique_fundamental_group_elements = []
    for element in current_known_fundamental_group_elements:
        if element not in unique_fundamental_group_elements:
            unique_fundamental_group_elements.append(element)

    current_known_fundamental_group_elements = unique_fundamental_group_elements


    unique_fundamental_group_elements = []
    for element in current_known_fundamental_group_elements:
        if element not in unique_fundamental_group_elements:
            unique_fundamental_group_elements.append(element)

    current_known_fundamental_group_elements = unique_fundamental_group_elements

    largest_and_smallest_eigenvals = []
    for i, element in enumerate(current_known_fundamental_group_elements):
        # print(i)
        eigenvalues = [np.float128(np.abs(x.evalf())) for x in list(element.eigenvals().keys())]
        largest_eigenval = np.max(eigenvalues)
        smallest_eigenval = np.min(eigenvalues)
        largest_and_smallest_eigenvals.append([largest_eigenval, smallest_eigenval])
    
    largest_and_smallest_eigenvals = np.array(largest_and_smallest_eigenvals)


    largest_and_smallest_eigenvals_visited = []
    for i, [alpha,beta] in enumerate(visited_generators_trace):
        eigenvalues = [np.float128(np.abs(x.evalf())) for x in list(beta.eigenvals().keys())]
        largest_eigenval = np.max(eigenvalues)
        smallest_eigenval = np.min(eigenvalues)
        largest_and_smallest_eigenvals_visited.append([largest_eigenval, smallest_eigenval])
    
    largest_and_smallest_eigenvals_visited = np.array(largest_and_smallest_eigenvals_visited)
            
    alpha,beta = visited_generators_trace[0]

    fig, ax = plt.subplots(figsize = (9, 6))

    traces = largest_and_smallest_eigenvals[:,0] + largest_and_smallest_eigenvals[:,1] + 1/(largest_and_smallest_eigenvals[:,0]*largest_and_smallest_eigenvals[:,1])
    lengths = np.log(largest_and_smallest_eigenvals[:,0]/largest_and_smallest_eigenvals[:,1])

    x_max = 1000
    
    ax.set_xlim(0,x_max)
    ax.scatter(traces, lengths, c='red')

    ax.set_ylim(0,25)

    def so21_function(x):
        return 2*np.log(1/2*(x**2)-1+1/2*np.sqrt(x**2*(x**2-4)))

    y = np.linspace(0,x_max, 10000)
    x = 3/(2**(2/3))*(np.exp(y)+1)**(2/3) * np.exp(-y/3)
    ax.plot(x,y, c='blue')
    ax.annotate(r'$l_{\text{max}}(\text{tr}(\gamma))$', xy=(x[170],y[170]),xytext=(x[170],y[170]+0.8), fontsize=50, c='blue',rotation=5)

    y = np.linspace(0,x_max, 10000)
    x = (np.exp(y)+2)* np.exp(-y/3)
    ax.plot(x,y, c='blue')
    ax.annotate(r'$l_{\text{min}}(\text{tr}(\gamma))$', xy=(x[95],y[95]),xytext=(x[95],y[95]-1.8), fontsize=50, c='blue',rotation=3)


    x = np.linspace(0, x_max, 1000)
    y = so21_function(np.sqrt(x+1))
    ax.plot(x,y, c='orange')
    ax.annotate(r'$l_{\text{SO}(2,1)}(\text{tr}(\gamma))$', xy=(x[550],y[550]),xytext=(x[550],y[550]+0.8), fontsize=22, c='orange',rotation=4)

    ax.scatter(traces, lengths, c='red',label=r'Random Elements From $\left\langle'+represent_matrix_as_latex(alpha_original) + ","+represent_matrix_as_latex(beta_original) +r'\right\rangle$')

    trace_traces_visited = []
    trace_lengths_visited = []
    for i, [alpha,beta] in enumerate(visited_generators_trace):
        trace_traces_visited.append(sp.trace(beta).evalf())
        trace_lengths_visited.append(calculate_geodesic_length(beta))
        
    length_traces_visited = []
    length_lengths_visited = []
    for i, [alpha,beta] in enumerate(visited_generators_length):
        length_traces_visited.append(sp.trace(beta).evalf())
        length_lengths_visited.append(calculate_geodesic_length(beta))

    peripheral = commutator(alpha_original, beta_original)
    ax.scatter(sp.trace(peripheral), calculate_geodesic_length(peripheral), c='orange',label='Peripheral Element')


    ax.plot(length_traces_visited, length_lengths_visited, c='cyan', label='Generalised Length Reduction Procedure Path')
    ax.scatter(length_traces_visited, length_lengths_visited, c='cyan')
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.plot(trace_traces_visited, trace_lengths_visited, c='green', linestyle='dashed', label='Generalised Trace Reduction Procedure Path')
    ax.scatter(trace_traces_visited, trace_lengths_visited, c='green')
    ax.scatter([length_traces_visited[-1]], [length_lengths_visited[-1]], c='blue', label='Length Reduction Procedure Terminated Representative')
    ax.scatter([trace_traces_visited[-1]], [trace_lengths_visited[-1]], c='purple', marker="v", label='Trace Reduction Procedure Terminated Representative')
    ax.set_xlabel(r'tr$(\gamma)$',  fontsize=50)
    ax.set_ylabel(r'$l(\gamma)$',  fontsize=50)
    ax.legend(fontsize=19,loc='upper left')
    plt.title(r"$\mathcal{X}$-Coordinates: " + str((t,t_prime, e01, e10, e12, e21, e20, e02)), fontsize=30)
    
    plt.show(block=blocking)
    plt.pause(0.01)
    if save_file_name:
        plt.savefig(f'./examples/{save_file_name}.pdf')


def rational_to_latex_fraction(rational):
    string_split = str(rational).rsplit("/")
    if len(string_split)>1:
        return "\\frac" + "{" + string_split[0] + "}{" + string_split[1] + "}"
    else:
        return str(rational)

def get_metrics(alpha_returned,beta_returned,expression, t,t_prime, e01, e10, e12, e21, e20, e02, latex=False):

    if not latex:
        return {'X-coordinates': str((t,t_prime, e01, e10, e12, e21, e20, e02)),
                "(A',B')": str(expression).replace("a","A").replace("b","B"),
                    "tr(B')": np.float64(sp.trace(beta_returned).evalf()),
                    "tr((B')^(-1))": np.float64(sp.trace(beta_returned.inv()).evalf()),
                    "tr(A')": np.float64(sp.trace(alpha_returned).evalf()),
                    "tr((A')^(-1))": np.float64(sp.trace(alpha_returned.inv()).evalf()),
                    "tr(A'B')": np.float64(sp.trace(alpha_returned*beta_returned).evalf()), 
                    "tr((A'B')^(-1))": np.float64(sp.trace((alpha_returned*beta_returned).inv()).evalf()),
                    "tr(A'(B')^(-1))": np.float64(sp.trace(alpha_returned*(beta_returned.inv())).evalf()),
                    "tr((A'(B')^(-1))^(-1))": np.float64(sp.trace((alpha_returned*(beta_returned.inv())).inv()).evalf()),
                    "tr([A',B'])": np.float64(sp.trace(commutator(alpha_returned,beta_returned)).evalf()),
                    "length(B')":calculate_geodesic_length(beta_returned),
                    "length(A')":calculate_geodesic_length(alpha_returned),
                    "length(A'B')": calculate_geodesic_length(alpha_returned*beta_returned),
                    "length(A'(B')^(-1))":calculate_geodesic_length(alpha_returned*(beta_returned.inv())),
                    "length([A',B'])":calculate_geodesic_length(commutator(alpha_returned,beta_returned))}
    
    else:
        coords = (t,t_prime, e01, e10, e12, e21, e20, e02)
        return {'$\mathcal{X}$-coordinates': "$" + str(tuple([rational_to_latex_fraction(x) for x in coords])).replace("'","").replace("\\\\","\\").replace("(","\\left(").replace(")","\\right)") + "$",
                "$(A',B')$": "$" + str(expression).replace("a","A").replace("b","B").replace("**","^").replace("*","") + "$", 
                    "$\\text{tr}(B')$": np.float64(sp.trace(beta_returned).evalf()),
                    "$\\text{tr}((B')^{-1})$": np.float64(sp.trace(beta_returned.inv()).evalf()),
                    "$\\text{tr}(A')$": np.float64(sp.trace(alpha_returned).evalf()),
                    "$\\text{tr}((A')^{-1})$": np.float64(sp.trace(alpha_returned.inv()).evalf()),
                    "$\\text{tr}(A'B')$": np.float64(sp.trace(alpha_returned*beta_returned).evalf()),
                    "$\\text{tr}((A'B')^{-1})$": np.float64(sp.trace((alpha_returned*beta_returned).inv()).evalf()),
                    "$\\text{tr}(A'(B')^{-1})$":np.float64(sp.trace(alpha_returned*(beta_returned.inv())).evalf()),
                    "$\\text{tr}((A'(B')^{-1})^{-1})$": np.float64(sp.trace((alpha_returned*(beta_returned.inv())).inv()).evalf()),
                    "$\\text{tr}([A',B'])$": np.float64(sp.trace(commutator(alpha_returned,beta_returned)).evalf()),
                    "$\\ell(B')$":calculate_geodesic_length(beta_returned),
                    "$\\ell(A')$":calculate_geodesic_length(alpha_returned),
                    "$\\ell(A'B')$": calculate_geodesic_length(alpha_returned*beta_returned),
                    "$\\ell(A'(B')^{-1})$":calculate_geodesic_length(alpha_returned*(beta_returned.inv())),
                    "$\\ell([A',B'])$":calculate_geodesic_length(commutator(alpha_returned,beta_returned))}
        


results_tr = pd.DataFrame()
results_l = pd.DataFrame()

latex = True

def canonical_generators(alpha,beta, length_function=calculate_geodesic_length, trace_function=sp.trace):
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
    



for example_function in give_all_examples():
    random_rationals = example_function()

    t = random_rationals[0]**3
    t_prime = random_rationals[1]**3
    e01 = random_rationals[2]**3
    e10 = random_rationals[3]**3
    e12 = random_rationals[4]**3
    e21 = random_rationals[5]**3
    e20 = random_rationals[6]**3
    e02 = random_rationals[7]**3


    print('X =',e01*e10*e12*e21*e20*e02)
    print('Y =',(t*t_prime)**3*e01*e10*e12*e21*e20*e02)

    alpha,beta = compute_translation_matrix_torus(t,t_prime, e01, e10, e12, e21, e20, e02, *random_rationals)

    print('A =', represent_matrix_as_latex(alpha))
    print('B =', represent_matrix_as_latex(beta))

    alpha, beta = canonical_generators(alpha,beta)
    # print(calculate_geodesic_length(alpha_prime)>= calculate_geodesic_length(beta_prime))
    # print(sp.trace(alpha_prime)>= sp.trace(beta_prime))


    alpha_returned, beta_returned, visited_generators_trace, expression = main_algorithm(alpha, beta, (lambda x : sp.trace(x),'trace'), [])

    metrics = get_metrics(alpha_returned, beta_returned, expression, t,t_prime, e01, e10, e12, e21, e20, e02, latex)
    for key in metrics.keys():
        metrics[key] = [metrics[key]]
    metrics["Objective"] = ["$\\text{tr}$" if latex else "tr"]
    metrics["Shorter End"] = ["Yes" if ("shorter" in str(example_function.__name__)) else "No"]    
    metrics["Label"] = [example_function.__name__.replace("generate","").replace("example","").replace("_"," ")[1:-1].replace("shorter ","").replace("longer ","").replace("end","").title()]
    results_tr = pd.concat([results_tr, pd.DataFrame(metrics)], ignore_index=True)


    print("----TRACE REDUCTION RESULTS----")
    print('X-coordinates:', (t,t_prime, e01, e10, e12, e21, e20, e02))
    print("Returned Generators (A',B'):", str(expression).replace("a","A").replace("b","B"))
    print("tr(A'):", sp.trace(alpha_returned).evalf(),"\n",
        "length(A'):",calculate_geodesic_length(alpha_returned),"\n",
        "tr(B'):",sp.trace(beta_returned).evalf(),"\n",
        "length(B')",calculate_geodesic_length(beta_returned),"\n",
        "tr(A'B'):", sp.trace(alpha_returned*beta_returned).evalf(),"\n",
        "length(A'B'):", calculate_geodesic_length(alpha_returned*beta_returned),"\n",
        "tr(A'(B')^(-1)):",sp.trace(alpha*(beta_returned.inv())).evalf(),"\n",
        "length(A'(B')^(-1)):",calculate_geodesic_length(alpha_returned*(beta_returned.inv())),"\n",
        "tr([A',B']):", sp.trace(commutator(alpha_returned,beta_returned)).evalf(),"\n",
        "length([A',B']):",calculate_geodesic_length(commutator(alpha_returned,beta_returned))
            )

    print('---------------------------------')

    alpha_returned, beta_returned, visited_generators_length, expression = main_algorithm(alpha, beta, (calculate_geodesic_length,'length'), [])
    
    
    metrics = get_metrics(alpha_returned, beta_returned, expression, t,t_prime, e01, e10, e12, e21, e20, e02, latex)
    for key in metrics.keys():
        metrics[key] = [metrics[key]]
    metrics["Objective"] = ["$\\ell$" if latex else "length"]
    metrics["Shorter End"] = ["Yes" if ("shorter" in str(example_function.__name__)) else "No"]    
    metrics["Label"] = [example_function.__name__.replace("generate","").replace("example","").replace("_"," ")[1:-1].replace("shorter ","").replace("longer ","").replace("end","").title()]
    results_l = pd.concat([results_l, pd.DataFrame(metrics)], ignore_index=True)


    print("----LENGTH REDUCTION RESULTS----")
    print('X-coordinates:', (t,t_prime, e01, e10, e12, e21, e20, e02))
    print("Returned Generators (A',B'):", str(expression).replace("a","A").replace("b","B"))
    print("tr(A'):", sp.trace(alpha_returned).evalf(),"\n",
        "length(A'):",calculate_geodesic_length(alpha_returned),"\n",
        "tr(B'):",sp.trace(beta_returned).evalf(),"\n",
        "length(B')",calculate_geodesic_length(beta_returned),"\n",
        "tr(A'B'):", sp.trace(alpha_returned*beta_returned).evalf(),"\n",
        "length(A'B'):", calculate_geodesic_length(alpha_returned*beta_returned),"\n",
        "tr(A'(B')^(-1)):",sp.trace(alpha_returned*(beta_returned.inv())).evalf(),"\n",
        "length(A'(B')^(-1)):",calculate_geodesic_length(alpha_returned*(beta_returned.inv())),"\n",
        "tr([A',B']):", sp.trace(commutator(alpha_returned,beta_returned)).evalf(),"\n",
        "length([A',B']):",calculate_geodesic_length(commutator(alpha_returned,beta_returned))
            )
    print('---------------------------------')

    #plot_eigenvalues_and_traces(alpha,beta, visited_generators_trace,visited_generators_length, num_distinct_random_group_elements=100, blocking=False)

results_tr = results_tr.round(2)
results_l = results_l.round(2)




columns_tr = np.array(results_tr.columns)
results_tr = pd.DataFrame(results_tr, columns=list(["Label","Shorter End", "Objective"])+list(columns_tr[:-3]))
results_tr.to_csv('./examples/results_tr.csv', sep=',', index=False)

columns_l = np.array(results_l.columns)
results_l = pd.DataFrame(results_l, columns=list(["Label","Shorter End", "Objective"])+list(columns_l[:-3]))
results_l.to_csv('./examples/results_l.csv', sep=',', index=False)



def colour_results(results):
    if not latex:
        starting_column_index = np.where(results.columns == "tr(B')")[0][0]
    else:
        starting_column_index = np.where(results.columns == "$\\text{tr}(B')$")[0][0]
    ending_column_index = len(results.columns)-1

    numbers_matrix = np.array(results)[:, starting_column_index:(ending_column_index+1)]

    numbers_matrix_trace = numbers_matrix[:, :9]
    numbers_matrix_length = numbers_matrix[:, 9:]

    colour_matrix = []

    colour_values = ["\\cellcolor{green!50}", "\\cellcolor{orange!50}", "\\cellcolor{red!25}", "\\cellcolor{red!50}"]

    for i in range(len(numbers_matrix_trace)):

        colours_matrix_row_trace = [""]*9
        colours_matrix_row_length = [""]*5

        number_row_trace = np.array(numbers_matrix_trace[i])
        number_row_length = np.array(numbers_matrix_length[i])

        number_row_trace[number_row_trace == 3] = np.inf
        number_row_length[number_row_length == 0] = np.inf

        sorted_trace_index = np.argsort(number_row_trace)
        sorted_length_index = np.argsort(number_row_length)

        for j in range(4):
            colours_matrix_row_trace[sorted_trace_index[j]] = colour_values[j]
            colours_matrix_row_length[sorted_length_index[j]] = colour_values[j]
        
        # check that peripheral is in smallest 3 list
        if colours_matrix_row_trace[-1] not in colour_values[:3]:
            colours_matrix_row_trace[sorted_trace_index[3]] = ""
        if colours_matrix_row_length[-1] not in colour_values[:3]:
            colours_matrix_row_length[sorted_length_index[3]] = ""

        colour_matrix.append(colours_matrix_row_trace + colours_matrix_row_length)
    results_copy = pd.DataFrame(results)

    for j in range(starting_column_index, ending_column_index+1):
        for i in range(len(results)):
            results_copy.iloc[i,j] = colour_matrix[i][j-starting_column_index] + str(results_copy.iloc[i,j])
    
    return results_copy


def display_results_latex():

    objectives = ['Trace', 'Length']
    directories = ['tr', 'l']

    for i in range(len(objectives)):
        results = pd.read_csv(f'./examples/results_{directories[i]}.csv')

        results = colour_results(results)

        column_names = results.columns.to_numpy()

        results = np.vstack([column_names, results.to_numpy()])

        results = results.T

        # results = np.hstack([results[:,:1],results[:, 7:]])

        # results[0,:] = np.array([results[0,i].replace("Hyperbolic Surface", "$\mathbb{H}^2$") for i in range(len(results[0,:]))])

        table_string = "\\begin{table}[!ht]\n\\centering\n\\begin{tabular}{" + "|l"*len(results[0,:]) + "|}\\hline\n"
        for row in results:
            table_string += " & ".join([str(x) for x in list(row)]) + " \\\\ \\hline \n"

        table_string+="\\end{tabular}\n\\caption{" + objectives[i] + " Minimization Examples}\\end{table}"
    
        print(table_string)

if latex:
    display_results_latex()


plt.show()
