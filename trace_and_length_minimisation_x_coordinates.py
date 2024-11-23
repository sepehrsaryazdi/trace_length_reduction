import numpy as np
import sympy as sp
from sympy import N
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings


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

def calculate_geodesic_length_high_precision(gamma,precision=1000): # slower implementation that maintains high precision and only rounds at the end
    gamma_eigenvalues = list(gamma.eigenvals())
    eigenvalues_to_precision = [N(gamma_eigenvalues[i],precision).as_real_imag()[0] for i in range(3)] # note that the real part is taken, the imaginary part is theoretically zero and negligible for a given rounded number.
    return np.float64(sp.log(sp.Max(*eigenvalues_to_precision)/sp.Min(*eigenvalues_to_precision)))

def calculate_geodesic_length(gamma):
    return calculate_geodesic_length_fast(gamma)


def get_bounds_algorithm(f):
    """
    Returns [k0,k1] such that f([k0,k1]) >= 0 if such values exist, otherwise it returns an empty array. Assumes f is strictly concave down and has a global maximiser.
    """
    middle_value = f(0)
    k1 = 0
    k0 = 0
    while f(k1) >= middle_value:
        k1 += 1
    while f(k0) >= middle_value:
        k0 -=1

    # at this point, [k0,k1] contains the maximiser of f. now may need to shrink or extend the domain until f(k) >= 0 <=> k in [k0,k1]
    
    k_vals = np.linspace(k0,k1,k1-k0+1, dtype=np.int64)
    f_k_vals = np.array([f(k) for k in k_vals])
    k_pos_vals = k_vals[f_k_vals > 0]
    if len(k_pos_vals) == 0:
        return [] # no values will be zero
    
    k0,k1 = np.min(k_pos_vals), np.max(k_pos_vals) # this set contains the minimiser and satisfies f(k) >= 0 for all k in [k0,k1]. Now can extend:
    finished_right = False
    k1_right = k1
    while not finished_right:
        if f(k1_right+1) >= 0:
            k1_right+=1
        else:
            finished_right = True
    
    finished_left = False
    k0_left = k0
    while not finished_left:
        if f(k0_left-1) >= 0:
            k0_left-=1
        else:
            finished_left = True
    
    k0,k1 = k0_left, k1_right
    return [k0,k1]


random_integers = [np.random.randint(1,10) for i in range(16)]

random_integers = [3,4, 4,5, 1,3, 1,2, 4,1, 2,9, 8,1, 7,6] # infinite volume example

random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]

t = random_rationals[0]**3
t_prime = random_rationals[1]**3
e01 = random_rationals[2]**3
e10 = random_rationals[3]**3
e12 = random_rationals[4]**3
e21 = random_rationals[5]**3
e20 = random_rationals[6]**3
e02 = random_rationals[7]**3

alpha,beta = compute_translation_matrix_torus(t,t_prime, e01, e10, e12, e21, e20, e02, *random_rationals)

def main_algorithm(alpha, beta, objective_function, visited_generators=[],expression=(sp.Symbol('a',commutative=False),sp.Symbol('b',commutative=False)), verbose=False):
    assert isinstance(alpha, sp.Matrix), "Error: alpha is not a sp.Matrix"
    assert isinstance(beta, sp.Matrix), "Error: beta is not a sp.Matrix"
 
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
    
    bounds = get_bounds_algorithm(f)

    if bounds:
        [k0_algorithm,k1_algorithm] = bounds
        k_vals =  np.linspace(k0_algorithm, k1_algorithm, k1_algorithm-k0_algorithm+1, dtype=np.int64)
        if len(k_vals):
            for k in k_vals:
                candidate_moves.append((k, X_k, f(k)))
    bounds = get_bounds_algorithm(g)
    if bounds:
        [k0_algorithm,k1_algorithm] = bounds
        k_vals = np.linspace(k0_algorithm, k1_algorithm, k1_algorithm-k0_algorithm+1, dtype=np.int64)
        if len(k_vals):
            for k in k_vals:
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


        alpha_prime, beta_prime, visited_generators,expression = main_algorithm(*move_function(alpha,beta, k), objective_function,visited_generators, expression)
 
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
    
    bounds = get_bounds_algorithm(f)

    if bounds:
        [k0_algorithm,k1_algorithm] = bounds
        k_vals =  np.linspace(k0_algorithm, k1_algorithm, k1_algorithm-k0_algorithm+1, dtype=np.int64)
        if len(k_vals):
            for k in k_vals:
                candidate_moves.append((k, X_k, f(k)))
    bounds = get_bounds_algorithm(g)
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




def plot_eigenvalues_and_traces(alpha, beta, visited_generators_trace, visited_generators_length, num_distinct_random_group_elements=100):
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

    ax.plot(length_traces_visited, length_lengths_visited, c='cyan', label='Generalised Length Reduction Procedure Path')
    ax.scatter(length_traces_visited, length_lengths_visited, c='cyan')
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.plot(trace_traces_visited, trace_lengths_visited, c='green', linestyle='dashed', label='Generalised Trace Reduction Procedure Path')
    ax.scatter(trace_traces_visited, trace_lengths_visited, c='green')
    ax.scatter([length_traces_visited[-1]], [length_lengths_visited[-1]], c='blue', label='Length Reduction Procedure Terminated Representative')
    ax.scatter([trace_traces_visited[-1]], [trace_lengths_visited[-1]], c='purple', label='Trace Reduction Procedure Terminated Representative')
    ax.set_xlabel(r'tr$(\gamma)$',  fontsize=50)
    ax.set_ylabel(r'$l(\gamma)$',  fontsize=50)
    ax.legend(fontsize=19,loc='upper left')
    plt.show()


print('A =', represent_matrix_as_latex(alpha))
print('B =', represent_matrix_as_latex(beta))


alpha_returned, beta_returned, visited_generators_trace, expression = main_algorithm(alpha, beta, lambda x : sp.trace(x), [])

alpha_length, beta_length = calculate_lengths(alpha_returned, beta_returned)


print("----TRACE MINIMISATION RESULTS----")
print('X coordinates:', (t,t_prime, e01, e10, e12, e21, e20, e02))
print('Returned Generators:', str(expression).replace("a","A").replace("b","B"))
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

alpha_returned, beta_returned, visited_generators_length, expression = main_algorithm(alpha, beta, calculate_geodesic_length, [])


print("----LENGTH MINIMISATION RESULTS----")
print('X coordinates:', (t,t_prime, e01, e10, e12, e21, e20, e02))
print('Returned Generators:', str(expression).replace("a","A").replace("b","B"))
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


plot_eigenvalues_and_traces(alpha,beta, visited_generators_trace,visited_generators_length, num_distinct_random_group_elements=100)
