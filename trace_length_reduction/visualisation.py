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


def create_latex_table_string(results):
    assert isinstance(results, ReductionResults), "Error: results must be of type ReductionResults."
    
    
    report = results.get_report()

    returned_expression = report["returned_expression"]
    alpha_returned, beta_returned = report["returned_generators"]
    coords = report["xcoords"]

    def convert_sympy_expr_to_latex(expression):
        return str(expression).replace("**","^").replace("*","").replace("(","{").replace(")","}")

    trace_and_length_results_latex = {
            '$\mathcal{X}$-coordinates': "$" + str(coords).replace("'","").replace("\\\\","\\").replace("(","\\left(").replace(")","\\right)") + "$",
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
    
    results_table = pd.DataFrame()

    for key in trace_and_length_results_latex.keys():
        trace_and_length_results_latex[key] = [trace_and_length_results_latex[key]]
    results_table = pd.concat([results_table, pd.DataFrame(trace_and_length_results_latex)], ignore_index=True)
    results_table = results_table.round(2)

    return results_table_to_latex(results_table, report["objective"])


class LengthTracePlot:
    """
    Class for initialising and visualising the length trace plot.
    """
    def __init__(self, trace_reduction_results, length_reduction_results, latex=True, title="Length Trace Plot"):
        assert isinstance(trace_reduction_results, ReductionResults), f"Error: {trace_reduction_results} must be an instance of ReductionResults."
        assert isinstance(length_reduction_results, ReductionResults), f"Error: {length_reduction_results} must be an instance of ReductionResults."

        self.latex = latex
        self.fontsize=20

        if self.latex:
            self.load_latex()

        self.create_figure(title)
        self.add_boundaries(self.ax)
        self.add_random_group_elements(self.ax, trace_reduction_results.get_report()["initial_generators"], trace_reduction_results.get_report()["visited_generators"], length_reduction_results.get_report()["visited_generators"])
        self.show_figure()
    
    def load_latex(self):
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
    

    def create_figure(self, title):
        self.fig, self.ax = plt.subplots(figsize = (9, 6))
        self.fig.canvas.manager.set_window_title(title)
        if self.latex:
            self.ax.set_xlabel(r'tr$(\gamma)$',  fontsize=self.fontsize)
            self.ax.set_ylabel(r'$l(\gamma)$',  fontsize=self.fontsize)
        else:
            self.ax.set_xlabel('tr(gamma)',  fontsize=self.fontsize)
            self.ax.set_ylabel('l(gamma)',  fontsize=self.fontsize)

        # self.ax.plot([1,2,3],[4,5,6])
        self.ax.set_title(title)
         

    def add_random_group_elements(self, ax, initial_generators, visited_generators_trace, visited_generators_length, num_distinct_random_group_elements=100):
        
        alpha,beta = initial_generators
        current_known_fundamental_group_elements = [commutator(alpha,beta)] + [alpha,beta] + [visited_generators_trace[i][0] for i in range(len(visited_generators_trace))] + [visited_generators_trace[i][1] for i in range(len(visited_generators_trace))] +  [visited_generators_length[i][0] for i in range(len(visited_generators_length))] + [visited_generators_length[i][1] for i in range(len(visited_generators_length))]
        
        # print((alpha.inv(),beta.inv()))
        # print(visited_generators_trace[0])
        # print([(calculate_geodesic_length(element[0]), calculate_geodesic_length(element[1])) for element in visited_generators_trace])

        # exit()

        unique_fundamental_group_elements = []
        for element in current_known_fundamental_group_elements:
            if element not in unique_fundamental_group_elements:
                unique_fundamental_group_elements.append(element)

        current_known_fundamental_group_elements = unique_fundamental_group_elements


        while len(current_known_fundamental_group_elements) < num_distinct_random_group_elements:
            use_beta = int(np.round(np.random.random_sample())) # if 0, use alpha, otherwise use beta
            power = (-1)**int(np.round(np.random.random_sample())) # if 0, use +1 power. otherwise use -1 power
            action = beta if use_beta else alpha
            action = action.inv() if power < 0 else action
            next_element_to_apply_action_index = np.random.choice([i for i in range(len(current_known_fundamental_group_elements))])
            next_element_to_apply_action = current_known_fundamental_group_elements[int(next_element_to_apply_action_index)]
            new_element = action * next_element_to_apply_action
            if new_element not in current_known_fundamental_group_elements:
                current_known_fundamental_group_elements.append(new_element)
                
        # unique_fundamental_group_elements = []
        # for element in current_known_fundamental_group_elements:
        #     if element not in unique_fundamental_group_elements:
        #         unique_fundamental_group_elements.append(element)

        # current_known_fundamental_group_elements = unique_fundamental_group_elements
        
        current_known_fundamental_group_elements_trace_length = np.array([[sp.trace(element).evalf(), calculate_geodesic_length(element)] for element in current_known_fundamental_group_elements])
        traces = current_known_fundamental_group_elements_trace_length[:,0]
        lengths = current_known_fundamental_group_elements_trace_length[:,1]

        ax.scatter(traces, lengths, c='red')

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
            ax.annotate(r'$l_{\text{SO}(2,1)}(\text{tr}(\gamma))$', xy=(x[text_position_index],y[text_position_index]),xytext=(x[text_position_index],y[text_position_index]+0.8), fontsize=self.fontsize, c='orange',rotation=4)
        else:
            ax.annotate('l_SO(2,1)(tr(gamma))', xy=(x[text_position_index],y[text_position_index]),xytext=(x[text_position_index],y[text_position_index]+0.8), fontsize=self.fontsize, c='orange',rotation=8)

        
        text_position_index = 8000

        y = np.linspace(0,1.81*y_max, 10000)
        x = trmax(y)
        ax.plot(x,y, c='blue')
        if self.latex:
            ax.annotate(r'$l_{\text{max}}(\text{tr}(\gamma))$', xy=(x[text_position_index],y[text_position_index]),xytext=(x[text_position_index],y[text_position_index]+1.2), fontsize=self.fontsize, c='blue',rotation=5)
        else:
            ax.annotate('l_max(tr(gamma))', xy=(x[text_position_index],y[text_position_index]),xytext=(x[text_position_index],y[text_position_index]+1.2), fontsize=self.fontsize, c='blue',rotation=10)

        text_position_index = 8300

        y = np.linspace(0,y_max, 10000)
        x = trmin(y)
        ax.plot(x,y, c='blue')
        if self.latex:
            ax.annotate(r'$l_{\text{min}}(\text{tr}(\gamma))$', xy=(x[text_position_index],y[text_position_index]),xytext=(x[text_position_index],y[text_position_index]-1.2), fontsize=self.fontsize, c='blue',rotation=3)
        else:
            ax.annotate('l_min(tr(gamma))', xy=(x[text_position_index],y[text_position_index]),xytext=(x[text_position_index],y[text_position_index]-1.2), fontsize=self.fontsize, c='blue',rotation=7)


        ax.set_xlim(0,trmax(1.81*y_max))
        ax.set_ylim(0,1.81*y_max)
       
        return ax
    


    def show_figure(self, timeout_sec=-1):
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

    def show_length_trace_plot(self, trace_reduction_results, length_reduction_results, title="Length Trace Plot"):
        assert isinstance(trace_reduction_results, ReductionResults), "Error: trace_reduction_results must be of class ReductionResults."
        assert isinstance(length_reduction_results, ReductionResults), "Error: length_reduction_results must be of class ReductionResults."
        length_trace_plot = LengthTracePlot(trace_reduction_results, length_reduction_results, latex=False, title=title)


    def process_and_display_inputs(self, x, event):
        
        x = self.process_inputs(x, event)
        
        trace_length_reduction_interface = TraceLengthReductionInterface(XCoords(x))

        trace_reduction_results = trace_length_reduction_interface.trace_reduction()
        length_reduction_results = trace_length_reduction_interface.length_reduction()
        
        self.display_results(trace_reduction_results, title=f"Trace Reduction Results (X-coords: {tuple(x)})")
        self.display_results(length_reduction_results, title=f"Length Reduction Results (X-coords: {tuple(x)})")
        self.show_length_trace_plot(trace_reduction_results, length_reduction_results, title=f"Length Trace Plot (X-coords: {tuple(x)})")
        

    def add_minimise_buttons(self, event, frame, default_inputs=XCoords([sp.Number(1)]*8)):
        x_coord_input_frame = tk.Frame(frame)
        x_coord_entries = [ttk.Entry(x_coord_input_frame, width=6) for i in range(8)]

        default_coords = default_inputs.get_coords()[0]
        [entry.insert(0,str(default_coords[i])) for i, entry in enumerate(x_coord_entries)]
        [entry.pack(side="left", ipady=5) for entry in x_coord_entries]
        x_coord_input_frame.pack(padx=5, pady=5)


        minimise_button = ttk.Button(x_coord_input_frame, text="Minimise")
        minimise_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        minimise_button.bind("<ButtonPress>", lambda event:self.process_and_display_inputs([x_coord_entries[i].get() for i in range(len(x_coord_entries))], event))

        def modify_inputs(new_inputs):
            assert isinstance(new_inputs, list) and len(new_inputs) == len(x_coord_entries), "Error: new_inputs must be a list of same length as x coordinates."
            for i in range(len(new_inputs)):
                x_coord_entries[i].delete(0,tk.END)
                x_coord_entries[i].insert(0, str(new_inputs[i]))

        cube_button = ttk.Button(x_coord_input_frame, text="Cube Inputs (Optional)")
        cube_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        cube_button.bind("<ButtonPress>", lambda event: self.cube_inputs([x_coord_entries[i].get() for i in range(len(x_coord_entries))], modify_inputs, event))

        randomise_button = ttk.Button(x_coord_input_frame, text="Randomise Inputs (Optional)")
        randomise_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        randomise_button.bind("<ButtonPress>", lambda event: self.randomise_inputs(modify_inputs, event))


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