from trace_length_reduction.reduction import ReductionResults, calculate_geodesic_length, commutator
from trace_length_reduction.reduction import XCoords, TraceLengthReductionInterface
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import tkinter as tk
from tkinter import ttk
import sympy as sp
from tkinter.scrolledtext import ScrolledText
from examples.example_generation_functions import give_all_examples
import pandas as pd


def represent_matrix_as_latex(M):
    return str(M).replace("Matrix","").replace("(", r"\begin{pmatrix} ").replace(")",r" \end{pmatrix}").replace("],",r" \\").replace(",",r" &").replace("[","").replace("]","")
    
def represent_matrix_as_text(M):
    return str(M).replace("Matrix","").replace("(", "").replace(")","").replace("],", "],\n")
    
def so21_function(x):
    return 2*np.log(1/2*(x**2)-1+1/2*np.sqrt(x**2*(x**2-4)))


def convert_input_to_rational(input):
    if '/' in input:
        input = input.rsplit('/')
        return sp.Number(input[0])/sp.Number(input[1])
    return sp.Number(input)



def colour_results(results):

    
    starting_column_index = np.where(results.columns == "$\\text{tr}(B')$")[0][0]
    ending_column_index = len(results.columns)-1

    numbers_matrix = np.array(results)[:, starting_column_index:(ending_column_index+1)]
    numbers_matrix_trace = numbers_matrix[:, :10]
    numbers_matrix_length = numbers_matrix[:, 10:]

    colour_matrix = []

    colour_values = ["\\cellcolor{green!50}", "\\cellcolor{orange!50}", "\\cellcolor{red!25}", "\\cellcolor{red!50}"]

    for i in range(len(numbers_matrix_trace)):

        colours_matrix_row_trace = [""]*10
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
        if (colours_matrix_row_trace[-1] not in colour_values[:3]) and (colours_matrix_row_trace[-2] not in colour_values[:3]):
            colours_matrix_row_trace[sorted_trace_index[3]] = ""
        if colours_matrix_row_length[-1] not in colour_values[:3]:
            colours_matrix_row_length[sorted_length_index[3]] = ""

        colour_matrix.append(colours_matrix_row_trace + colours_matrix_row_length)
    results_copy = pd.DataFrame(results)

    for j in range(starting_column_index, ending_column_index+1):
        for i in range(len(results)):
            results_copy.iloc[i,j] = colour_matrix[i][j-starting_column_index] + str(results_copy.iloc[i,j])
    
    return results_copy



def results_table_to_latex(results_table, objective):

    results_table = colour_results(results_table)

    column_names = results_table.columns.to_numpy()

    results_table = np.vstack([column_names, results_table.to_numpy()])

    results_table = results_table.T

    # results = np.hstack([results[:,:1],results[:, 7:]])

    # results[0,:] = np.array([results[0,i].replace("Hyperbolic Surface", "$\mathbb{H}^2$") for i in range(len(results[0,:]))])

    table_string = "\\begin{table}[!ht]\n\\centering\n\\begin{tabular}{" + "|l"*len(results_table[0,:]) + "|}\\hline\n"
    for row in results_table:
        table_string += " & ".join([str(x) for x in list(row)]) + " \\\\ \\hline \n"

    table_string+="\\end{tabular}\n\\caption{" + str(objective).capitalize() + " Minimization Examples}\\end{table}"

    return table_string

def convert_sympy_expr_to_latex(expression):
    return str(expression).replace("**","^").replace("*","").replace("(","{").replace(")","}")


def generate_trace_and_length_results_latex(coords, returned_expression, alpha_returned, beta_returned):
    return {'$\mathcal{X}$-coordinates': "$" + str(coords).replace("'","").replace("\\\\","\\").replace("(","\\left(").replace(")","\\right)") + "$",
            "$(A',B')$": "$(" + convert_sympy_expr_to_latex(returned_expression[0]) + ", " + convert_sympy_expr_to_latex(returned_expression[1])+ ")$", 
                "$\\text{tr}(B')$": np.float64(sp.trace(beta_returned).evalf()),
                "$\\text{tr}((B')^{-1})$": np.float64(sp.trace(beta_returned.inv()).evalf()),
                "$\\text{tr}(A')$": np.float64(sp.trace(alpha_returned).evalf()),
                "$\\text{tr}((A')^{-1})$": np.float64(sp.trace(alpha_returned.inv()).evalf()),
                "$\\text{tr}(A'B')$": np.float64(sp.trace(alpha_returned*beta_returned).evalf()),
                "$\\text{tr}((A'B')^{-1})$": np.float64(sp.trace((alpha_returned*beta_returned).inv()).evalf()),
                "$\\text{tr}(A'(B')^{-1})$":np.float64(sp.trace(alpha_returned*(beta_returned.inv())).evalf()),
                "$\\text{tr}((A'(B')^{-1})^{-1})$": np.float64(sp.trace((alpha_returned*(beta_returned.inv())).inv()).evalf()),
                "$\\text{tr}([A',B'])$": np.float64(sp.trace(commutator(alpha_returned,beta_returned)).evalf()),
                "$\\text{tr}([A',B']^{-1})$": np.float64(sp.trace(commutator(alpha_returned,beta_returned).inv()).evalf()),
                "$\\ell(B')$":calculate_geodesic_length(beta_returned),
                "$\\ell(A')$":calculate_geodesic_length(alpha_returned),
                "$\\ell(A'B')$": calculate_geodesic_length(alpha_returned*beta_returned),
                "$\\ell(A'(B')^{-1})$":calculate_geodesic_length(alpha_returned*(beta_returned.inv())),
                "$\\ell([A',B'])$":calculate_geodesic_length(commutator(alpha_returned,beta_returned))}

def create_latex_table_string(results):
    assert isinstance(results, ReductionResults), "Error: results must be of type ReductionResults."
    
    
    report = results.get_report()

    returned_expression = report["returned_expression"]
    alpha_returned, beta_returned = report["returned_generators"]
    coords = report["xcoords"]

    trace_and_length_results_latex = generate_trace_and_length_results_latex(coords, returned_expression, alpha_returned, beta_returned)
    
    results_table = pd.DataFrame()

    for key in trace_and_length_results_latex.keys():
        trace_and_length_results_latex[key] = [trace_and_length_results_latex[key]]
    results_table = pd.concat([results_table, pd.DataFrame(trace_and_length_results_latex)], ignore_index=True)
    results_table = results_table.round(2)

    return results_table_to_latex(results_table, report["objective"])


class TraceLengthElements:
    """
    Class for visualising elements found during reduction, along with their length and traces.
    """
    def __init__(self, elements:list,elements_sym:np.ndarray, title="Length Trace"):
        assert isinstance(elements, list), "Error: elements must be of type list."
        for element in elements:
            assert isinstance(element, sp.Matrix), "Error: element must be of type sp.Matrix"
        

        assert isinstance(elements_sym, np.ndarray), "Error: elements must be of type np.ndarray."
        for element in elements_sym:
            assert isinstance(element, sp.Expr), "Error: element must be of type sp.Expr"
        

        trace_length_elements_window = tk.Toplevel()
        trace_length_elements_window.wm_title(title)

        canvas = tk.Canvas(trace_length_elements_window)
        scrollbar = ttk.Scrollbar(trace_length_elements_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", ipadx=200, expand=True)
        scrollbar.pack(side="right", fill="y", ipady=300)


        # def copy_to_clipboard(text):
        #     trace_length_elements_window.clipboard_clear()
        #     trace_length_elements_window.clipboard_append(text)

        # trace_length_elements_frame = tk.Frame(scrollable_frame)
        # copy_trace_length_elements_to_clipboard_button = ttk.Button(trace_length_elements_frame, text="Copy Output")
        # copy_trace_length_elements_to_clipboard_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        # copy_trace_length_elements_to_clipboard_button.bind("<ButtonPress>", lambda event : copy_to_clipboard(text_result_string))

        # text_trace_length_elements = tk.Text(trace_length_elements_frame)
        # text_trace_length_elements.pack(side="left", ipady=150)
        # text_trace_length_elements.bind("<KeyPress>", lambda x:x)
        # # text.insert("end", "Hello"+"\n"*40+"World.")
        # text_trace_length_elements.insert("end", text_result_string)

        # trace_length_elements_frame.pack()


        # table_string = create_latex_table_string(trace_length_elements)


        # table_frame = tk.Frame(scrollable_frame)
        # copy_table_to_clipboard_button = ttk.Button(table_frame, text="Copy Output")
        # copy_table_to_clipboard_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        # copy_table_to_clipboard_button.bind("<ButtonPress>", lambda event : copy_to_clipboard(table_string))
        # text_table = tk.Text(table_frame)
        # text_table.pack(side="left", ipady=150)
        # text_table.bind("<KeyPress>", lambda x:x)
        # # text.insert("end", "Hello"+"\n"*40+"World.")
        # text_table.insert("end", table_string)

        # table_frame.pack()


class LengthTracePlot:
    """
    Class for initialising and visualising the length trace plot.
    """
    def __init__(self, reduction_results, latex=True, title="Length Trace Plot"):
        assert isinstance(reduction_results, ReductionResults), f"Error: {reduction_results} must be an instance of ReductionResults."

        self.latex = latex
        self.fontsize=20

        coords = reduction_results.get_report()["xcoords"]
        returned_expression = reduction_results.get_report()["returned_expression"]
        objective = reduction_results.get_objective()
        if self.latex:
            self.load_latex()
            plot_title =f"{objective.capitalize()}" + r" Reduction Length Trace Plot ($\mathcal{X}$-coordinates: $" + str(coords) + r")$"
        else:
            plot_title = f"{objective.capitalize()}" + " Reduction Length Trace Plot (X-coordinates: " + str(coords) + ")"
        window_title = f"{objective.capitalize()}" + " Reduction Length Trace Plot (X-coordinates: " + str(coords) + ")"
        self.create_figure(plot_title, window_title)
        self.add_boundaries(self.ax)
        self.add_random_group_elements(self.ax, coords, reduction_results.get_objective(), returned_expression, reduction_results.get_report()["initial_generators"],reduction_results.get_report()["returned_generators"], reduction_results.get_report()["visited_generators"], reduction_results.get_report()["expressions"], title=window_title)
        self.show_figure()
    
    def load_latex(self):
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
    

    def create_figure(self, plot_title, window_title):
        self.fig, self.ax = plt.subplots(figsize = (9, 6))
        self.fig.canvas.manager.set_window_title(window_title)
        if self.latex:
            self.ax.set_xlabel(r'tr$(\gamma)$',  fontsize=self.fontsize)
            self.ax.set_ylabel(r'$\ell(\gamma)$',  fontsize=self.fontsize)
        else:
            self.ax.set_xlabel('tr(gamma)',  fontsize=self.fontsize)
            self.ax.set_ylabel('l(gamma)',  fontsize=self.fontsize)

        self.ax.set_title(plot_title)
    
    def create_random_group_elements(self, initial_generators, returned_generators, visited_generators, num_distinct_random_group_elements=200):
        alpha,beta = initial_generators
        alpha_returned, beta_returned = returned_generators
        peripheral = commutator(alpha_returned,beta_returned)
        AB = alpha_returned*beta_returned
        AB_inv = alpha_returned*(beta_returned**(-1))
        current_known_fundamental_group_elements = [peripheral, AB, AB_inv] + [alpha,beta] + [visited_generators[i][0] for i in range(len(visited_generators))] + [visited_generators[i][1] for i in range(len(visited_generators))]
        unique_fundamental_group_elements = []
        for element in current_known_fundamental_group_elements:
            if element not in unique_fundamental_group_elements:
                unique_fundamental_group_elements.append(element)

        current_known_fundamental_group_elements = unique_fundamental_group_elements


        while len(current_known_fundamental_group_elements) < num_distinct_random_group_elements:
            use_beta = int(np.round(np.random.random_sample())) # if 0, use alpha, otherwise use beta
            power = (-1)**int(np.round(np.random.random_sample())) # if 0, use +1 power. otherwise use -1 power
            action = beta if use_beta else alpha
            action = action**(-1) if power < 0 else action
            next_element_to_apply_action_index = np.random.choice([i for i in range(len(current_known_fundamental_group_elements))])
            next_element_to_apply_action = current_known_fundamental_group_elements[int(next_element_to_apply_action_index)]
            new_element = action * next_element_to_apply_action
            if new_element not in current_known_fundamental_group_elements:
                current_known_fundamental_group_elements.append(new_element)

        return current_known_fundamental_group_elements

    def add_random_group_elements(self, ax, coords, objective, returned_expression, initial_generators, returned_generators, visited_generators, expressions, num_distinct_random_group_elements=200, title="Length Trace Plot"):
        
        alpha,beta = initial_generators
        alpha_returned,beta_returned = returned_generators

        peripheral = commutator(alpha_returned,beta_returned)
        
        AB = alpha_returned*beta_returned

        AB_inv = alpha_returned*(beta_returned.inv())

        current_known_fundamental_group_elements = self.create_random_group_elements(initial_generators, returned_generators, visited_generators, num_distinct_random_group_elements)
        current_known_fundamental_group_elements_sym = self.create_random_group_elements((sp.Symbol("A",commutative=False),sp.Symbol("B",commutative=False)), returned_expression, expressions, num_distinct_random_group_elements)

  

        current_known_fundamental_group_elements_trace_length = np.array([[sp.trace(element).evalf(), calculate_geodesic_length(element)] for element in current_known_fundamental_group_elements])
        traces = current_known_fundamental_group_elements_trace_length[:,0]
        lengths = current_known_fundamental_group_elements_trace_length[:,1]

        length_trace_values = np.array([(ai, bi) for ai, bi in zip(lengths, traces)],
             dtype = np.dtype([('x', float), ('y', float)]))
        
        current_known_fundamental_group_elements_sym = np.array(current_known_fundamental_group_elements_sym)
        # current_known_fundamental_group_elements_sym = current_known_fundamental_group_elements_sym[np.argsort(length_trace_values)]

        # print(current_known_fundamental_group_elements_sym)

        # TraceLengthElements(current_known_fundamental_group_elements,current_known_fundamental_group_elements_sym, title=title.replace("Plot","Found Elements"))

        if self.latex:
            ax.scatter(traces, lengths, c='red',label=r'Random Elements From $\left\langle'+represent_matrix_as_latex(alpha) + ","+represent_matrix_as_latex(beta) +r'\right\rangle$')
        else:
            ax.scatter(traces, lengths, c='red',label='Random Elements From \n<'+represent_matrix_as_text(alpha) + ",\n"+represent_matrix_as_text(beta) + ">")

        visited_trace_length_1st = np.array([[sp.trace(element[0]).evalf(), calculate_geodesic_length(element[0])] for element in visited_generators])

        ax.plot(visited_trace_length_1st[:,0], visited_trace_length_1st[:,1], c='cyan', label=f'Generalised {objective.capitalize()} Reduction 1st Generator Path')
        ax.scatter(visited_trace_length_1st[:,0], visited_trace_length_1st[:,1], c='cyan')
        if self.latex:
            ax.scatter([visited_trace_length_1st[-1][0]], [visited_trace_length_1st[-1][1]], c='darkturquoise', label=r"$A'$")
        else:
            ax.scatter([visited_trace_length_1st[-1][0]], [visited_trace_length_1st[-1][1]], c='darkturquoise', label="A'")
            
        visited_trace_length_2nd = np.array([[sp.trace(element[1]).evalf(), calculate_geodesic_length(element[1])] for element in visited_generators])

        ax.plot(visited_trace_length_2nd[:,0], visited_trace_length_2nd[:,1], linestyle='dashed', c='green', label=f'Generalised {objective.capitalize()} Reduction 2nd Generator Path')
        ax.scatter(visited_trace_length_2nd[:,0], visited_trace_length_2nd[:,1], c='green', marker="v")
        if self.latex:
            ax.scatter([visited_trace_length_2nd[-1][0]], [visited_trace_length_2nd[-1][1]], marker="v",c='purple', label=r"$B'$")
        else:
            ax.scatter([visited_trace_length_2nd[-1][0]], [visited_trace_length_2nd[-1][1]], marker="v",c='purple', label="B'")

        alpha_returned,beta_returned = returned_generators
        if self.latex:
            ax.scatter(sp.trace(peripheral), calculate_geodesic_length(peripheral), c='orange',label=r"$[A',B']$", marker="s",s=15)
        else:
            ax.scatter(sp.trace(peripheral), calculate_geodesic_length(peripheral), c='orange',label="[A',B']", marker="s",s=15)

        if self.latex:    
            ax.scatter(sp.trace(AB), calculate_geodesic_length(AB), c='black',label=r"$A'B'$", marker="p",s=15)
        else:
            ax.scatter(sp.trace(AB), calculate_geodesic_length(AB), c='black',label="A'B'", marker="p",s=15)
        
        
        if self.latex:    
            ax.scatter(sp.trace(AB_inv), calculate_geodesic_length(AB_inv), c='darkblue',label=r"$A'(B')^{-1}$", marker="2")
        else:
            ax.scatter(sp.trace(AB_inv), calculate_geodesic_length(AB_inv), c='darkblue',label="A'(B')" + "**" + "(-1)", marker="2")
        


        # current_known_fundamental_group_elements_rel_termination = self.create_random_group_elements(returned_generators, initial_generators, visited_generators, num_distinct_random_group_elements)
        # current_known_fundamental_group_elements_rel_termination_sym = self.create_random_group_elements((sp.Symbol("A_p",commutative=False),sp.Symbol("B_p",commutative=False)), (sp.Symbol("A_p",commutative=False),sp.Symbol("B_p",commutative=False)), expressions, num_distinct_random_group_elements)

        

        if objective == "length":
            
            A, B = returned_generators
            A_s, B_s = (sp.Symbol("A'",commutative=False),sp.Symbol("B'",commutative=False))
            
            elements_matrices = []
            elements_symbols = []

            for i in range(-10, 10):
                elements_matrices.append(A*(B**i))
                elements_symbols.append(A_s*(B_s**i))

            f, ax = plt.subplots(figsize = (9, 6))
            ticks = [r"$" +str(s).replace("**","^").replace("*","").replace("(","{").replace(")","}").replace("A'","(A')").replace("B'","(B')") + r"$" for s in elements_symbols]
            ax.plot(ticks, [calculate_geodesic_length(e) for e in elements_matrices])
            ax.set_xticklabels(ticks, rotation=45, ha='right',fontsize=self.fontsize - 10)
            ax.set_ylabel(r"$\ell(\gamma)$",fontsize=self.fontsize)
            ax.set_xlabel("$\gamma$",fontsize=self.fontsize)
            ax.set_title(title.replace("Length Trace Plot", "Length by Element"),fontsize=self.fontsize)
            plt.show()

        return ax

    def add_boundaries(self, ax, y_max=10):
        assert isinstance(ax, plt.Axes), "Error: ax must be a plt.Axes object."
        
        def trmax(l):
            return 3/(2**(2/3))*(np.exp(l)+1)**(2/3) * np.exp(-l/3)
        def trmin(l):
            return (np.exp(l)+2)* np.exp(-l/3)
        
        text_position_index = 2500


        x = np.linspace(0, trmin(y_max), 10000)
        y = so21_function(np.sqrt(x+1))
        ax.plot(x,y, c='orange')
        if self.latex:
            ax.annotate(r'$\ell_{\text{SO}(2,1)}(\text{tr}(\gamma))$', xy=(x[text_position_index],y[text_position_index]),xytext=(x[text_position_index],y[text_position_index]+0.8), fontsize=self.fontsize, c='orange',rotation=9)
        else:
            ax.annotate('l_SO(2,1)(tr(gamma))', xy=(x[text_position_index],y[text_position_index]),xytext=(x[text_position_index],y[text_position_index]+0.8), fontsize=self.fontsize, c='orange',rotation=9)

        
        text_position_index = 8000

        y = np.linspace(0,1.81*y_max, 10000)
        x = trmax(y)
        ax.plot(x,y, c='blue')
        if self.latex:
            ax.annotate(r'$\ell_{\text{max}}(\text{tr}(\gamma))$', xy=(x[text_position_index],y[text_position_index]),xytext=(x[text_position_index],y[text_position_index]+1.2), fontsize=self.fontsize, c='blue',rotation=10)
        else:
            ax.annotate('l_max(tr(gamma))', xy=(x[text_position_index],y[text_position_index]),xytext=(x[text_position_index],y[text_position_index]+1.2), fontsize=self.fontsize, c='blue',rotation=10)

        text_position_index = 8300

        y = np.linspace(0,y_max, 10000)
        x = trmin(y)
        ax.plot(x,y, c='blue')
        if self.latex:
            ax.annotate(r'$\ell_{\text{min}}(\text{tr}(\gamma))$', xy=(x[text_position_index],y[text_position_index]),xytext=(x[text_position_index],y[text_position_index]-1.2), fontsize=self.fontsize, c='blue',rotation=7)
        else:
            ax.annotate('l_min(tr(gamma))', xy=(x[text_position_index],y[text_position_index]),xytext=(x[text_position_index],y[text_position_index]-1.2), fontsize=self.fontsize, c='blue',rotation=7)


        ax.set_xlim(0,trmax(1.81*y_max))
        ax.set_ylim(0,1.81*y_max)

       
        return ax
    

        
    


    def show_figure(self, timeout_sec=0.01):
        self.ax.legend(fontsize=int(self.fontsize/2),loc='lower right')
        self.fig.show()
        plt.pause(timeout_sec)



class Menu:
    """
    Class for visualising menu for seeing examples or minimising a set of X-coordinates.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Trace Length Reduction Algorithm Menu")
        self.root.geometry("520x120")
        self.menubar = tk.Menu(self.root)        
        self.app = tk.Frame(self.root)
        self.button_frame = ttk.Frame(self.root)    
        self.button_frame.pack(side="top", pady=(20,0))
        try:
            photo = tk.PhotoImage(file = 'trace_length_reduction/icon.png')
            self.root.wm_iconphoto(False, photo)
        except:
            pass

        self.examples_button= ttk.Button(self.button_frame, text="Examples")
        self.examples_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        self.examples_button.bind("<ButtonPress>", lambda event : self.show_examples(event))

        self.minimise_button= ttk.Button(self.button_frame, text="Minimise X-coordinates")
        self.minimise_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        self.minimise_button.bind("<ButtonPress>", lambda event : self.show_minimise_window(event))

    def show_examples(self ,event):
        
        examples_window = tk.Toplevel()
        examples_window.resizable(width=False, height=False)
        examples_window.wm_title("Examples")
        examples_window_text = tk.Label(examples_window,
                                            text="Please select an example below.")
        examples_window_text.pack(padx=5, pady=5)


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
            self.add_minimise_buttons(event,examples_window,default_inputs=XCoords([t,t_prime, e01, e10, e12, e21, e20, e02]))


    def display_results(self, results, title="Results"):
        assert isinstance(results, ReductionResults), "Error: results must be of type ReductionResults."

        results_window = tk.Toplevel()
        results_window.wm_title(title)

        canvas = tk.Canvas(results_window)
        scrollbar = ttk.Scrollbar(results_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", ipadx=200, expand=True)
        scrollbar.pack(side="right", fill="y", ipady=300)


        text_result_string = str(results.get_report())

        def copy_to_clipboard(text):
            results_window.clipboard_clear()
            results_window.clipboard_append(text)

        results_frame = tk.Frame(scrollable_frame)
        copy_results_to_clipboard_button = ttk.Button(results_frame, text="Copy Output")
        copy_results_to_clipboard_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        copy_results_to_clipboard_button.bind("<ButtonPress>", lambda event : copy_to_clipboard(text_result_string))

        text_results = tk.Text(results_frame)
        text_results.pack(side="left", ipady=150)
        text_results.bind("<KeyPress>", lambda x:x)
        # text.insert("end", "Hello"+"\n"*40+"World.")
        text_results.insert("end", text_result_string)

        results_frame.pack()


        table_string = create_latex_table_string(results)


        table_frame = tk.Frame(scrollable_frame)
        copy_table_to_clipboard_button = ttk.Button(table_frame, text="Copy Output")
        copy_table_to_clipboard_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        copy_table_to_clipboard_button.bind("<ButtonPress>", lambda event : copy_to_clipboard(table_string))
        text_table = tk.Text(table_frame)
        text_table.pack(side="left", ipady=150)
        text_table.bind("<KeyPress>", lambda x:x)
        # text.insert("end", "Hello"+"\n"*40+"World.")
        text_table.insert("end", table_string)

        table_frame.pack()

    def cube_inputs(self, x, return_func, event):
        x = self.process_inputs(x,event)
        return_func([xi**3 for xi in x])
    
    def randomise_inputs(self, return_func, event):
        random_integers = [np.random.randint(1,10) for i in range(16)]
        random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
        return_func(random_rationals)

    def process_inputs(self, x, event):
        x = [convert_input_to_rational(xi) for xi in x]
        return x

    def show_length_trace_plot(self, reduction_results, latex_rendering=False):
        assert isinstance(reduction_results, ReductionResults), "Error: trace_reduction_results must be of class ReductionResults."
        length_trace_plot = LengthTracePlot(reduction_results, latex=latex_rendering)


    def process_and_display_inputs(self, x, event, latex_rendering=False):
        
        x = self.process_inputs(x, event)
    
        trace_length_reduction_interface = TraceLengthReductionInterface(XCoords(x))

        trace_reduction_results = trace_length_reduction_interface.trace_reduction()
        length_reduction_results = trace_length_reduction_interface.length_reduction()
        
        self.display_results(trace_reduction_results, title=f"Trace Reduction Results (X-coords: {tuple(x)})")
        self.display_results(length_reduction_results, title=f"Length Reduction Results (X-coords: {tuple(x)})")
        
        self.show_length_trace_plot(trace_reduction_results, latex_rendering=latex_rendering)
        self.show_length_trace_plot(length_reduction_results, latex_rendering=latex_rendering)
        

    def add_minimise_buttons(self, event, frame, default_inputs=XCoords([sp.Number(1)]*8)):
        
        all_elements_frame = tk.Frame(frame)
        
        x_coord_frame = tk.Frame(all_elements_frame)
        
        x_coord_tag_frame = tk.Frame(x_coord_frame)
        x_coord_tags = ["t"," t'", "e01", "e10", "e12", "e21", "e20", "e02"]
        x_coord_tag_texts = [tk.Label(x_coord_tag_frame, text=tag) for tag in x_coord_tags]
        [text.pack(side="left",ipadx=12, ipady=3) for text in x_coord_tag_texts]
        x_coord_tag_frame.pack()

        x_coord_input_frame = tk.Frame(x_coord_frame)
        x_coord_entries = [ttk.Entry(x_coord_input_frame, width=6) for i in range(8)]
        default_coords = default_inputs.get_coords()[0]
        [entry.insert(0,str(default_coords[i])) for i, entry in enumerate(x_coord_entries)]
        [entry.pack(side="left", ipady=5) for entry in x_coord_entries]
        x_coord_input_frame.pack(padx=5, pady=5)

        x_coord_frame.pack(side="left")

        buttons_frame = tk.Frame(all_elements_frame)

        minimise_button = ttk.Button(buttons_frame, text="Minimise")
        minimise_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        minimise_button.bind("<ButtonPress>", lambda event:self.process_and_display_inputs([x_coord_entries[i].get() for i in range(len(x_coord_entries))], event, latex_rendering=bool(latex_state.get())))

        def modify_inputs(new_inputs):
            assert isinstance(new_inputs, list) and len(new_inputs) == len(x_coord_entries), "Error: new_inputs must be a list of same length as x coordinates."
            for i in range(len(new_inputs)):
                x_coord_entries[i].delete(0,tk.END)
                x_coord_entries[i].insert(0, str(new_inputs[i]))

        latex_state = tk.IntVar(value=1)
        latex_check_button = tk.Checkbutton(buttons_frame, text = "LaTeX Rendering", 
                    variable = latex_state, 
                    onvalue = 1, 
                    offvalue = 0, 
                    height = 2, 
                    width = 10)
        latex_check_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        latex_check_button.select()


        cube_button = ttk.Button(buttons_frame, text="Cube Inputs (Optional)")
        cube_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        cube_button.bind("<ButtonPress>", lambda event: self.cube_inputs([x_coord_entries[i].get() for i in range(len(x_coord_entries))], modify_inputs, event))

        randomise_button = ttk.Button(buttons_frame, text="Randomise Inputs (Optional)")
        randomise_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        randomise_button.bind("<ButtonPress>", lambda event: self.randomise_inputs(modify_inputs, event))

        buttons_frame.pack(side="left")

        all_elements_frame.pack()

    def show_minimise_window(self, event, default_inputs=XCoords([sp.Number(1)]*8)):
        assert isinstance(default_inputs, XCoords), "Error: Default inputs must be of class XCoords."
        
        minimise_window = tk.Toplevel()
        minimise_window.resizable(width=False, height=False)
        minimise_window.wm_title("Minimise X-coordinates")
        minimise_window_text = tk.Label(minimise_window,
                                            text="Please enter X-coordinates below, then press Minimise to generate reports.")
        minimise_window_text.pack(padx=5, pady=5)

        self.add_minimise_buttons(event, minimise_window, default_inputs)



    def show(self):
        self.app.mainloop()