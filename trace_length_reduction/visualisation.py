from trace_length_reduction.reduction import ReductionResults
from trace_length_reduction.reduction import XCoords, TraceLengthReductionInterface
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import tkinter as tk
from tkinter import ttk
import sympy as sp
from tkinter.scrolledtext import ScrolledText
from examples.example_generation_functions import give_all_examples


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
                    "tr([A',B']^{-1})": np.float64(sp.trace(commutator(alpha_returned,beta_returned).inv()).evalf()),
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
                    "$\\text{tr}([A',B']^{-1})$": np.float64(sp.trace(commutator(alpha_returned,beta_returned).inv()).evalf()),
                    "$\\ell(B')$":calculate_geodesic_length(beta_returned),
                    "$\\ell(A')$":calculate_geodesic_length(alpha_returned),
                    "$\\ell(A'B')$": calculate_geodesic_length(alpha_returned*beta_returned),
                    "$\\ell(A'(B')^{-1})$":calculate_geodesic_length(alpha_returned*(beta_returned.inv())),
                    "$\\ell([A',B'])$":calculate_geodesic_length(commutator(alpha_returned,beta_returned))}



def so21_function(x):
    return 2*np.log(1/2*(x**2)-1+1/2*np.sqrt(x**2*(x**2-4)))

class LengthTracePlot:
    """
    Class for initialising and visualising the length trace plot.
    """
    def __init__(self, trace_reduction_results, length_reduction_results, latex=True):
        assert isinstance(trace_reduction_results, ReductionResults), f"Error: {trace_reduction_results} must be an instance of ReductionResults."
        assert isinstance(length_reduction_results, ReductionResults), f"Error: {length_reduction_results} must be an instance of ReductionResults."

        

        print(trace_reduction_results.get_report())

        if latex:
            self.load_latex()

        self.create_figure()

        self.show_figure()
    
    def load_latex(self):
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
    

    def create_figure(self):
        self.fig, self.ax = plt.subplots(figsize = (9, 6))
        self.ax.plot([1,2,3],[4,5,6])

    def add_boundaries(self, ax, x_max=1000, y_max=1000):
        assert isinstance(ax, plt.Axes), "Error: ax must be a plt.Axes object."
        
        y = np.linspace(0,x_max, 10000)
        x = 3/(2**(2/3))*(np.exp(y)+1)**(2/3) * np.exp(-y/3)
        ax.plot(x,y, c='blue')
        ax.annotate(r'$l_{\text{max}}(\text{tr}(\gamma))$', xy=(x[170],y[170]),xytext=(x[170],y[170]+0.8), fontsize=50, c='blue',rotation=5)

        y = np.linspace(0,y_max, 10000)
        x = (np.exp(y)+2)* np.exp(-y/3)
        ax.plot(x,y, c='blue')
        ax.annotate(r'$l_{\text{min}}(\text{tr}(\gamma))$', xy=(x[95],y[95]),xytext=(x[95],y[95]-1.8), fontsize=50, c='blue',rotation=3)

        x = np.linspace(0, x_max, 1000)
        y = so21_function(np.sqrt(x+1))
        ax.plot(x,y, c='orange')
        ax.annotate(r'$l_{\text{SO}(2,1)}(\text{tr}(\gamma))$', xy=(x[550],y[550]),xytext=(x[550],y[550]+0.8), fontsize=22, c='orange',rotation=4)

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

        text_results = str(results.get_report())

        def copy_to_clipboard(text):
            results_window.clipboard_clear()
            results_window.clipboard_append(text)
        results_frame = tk.Frame(results_window)
        copy_to_clipboard_button = ttk.Button(results_frame, text="Copy Output")
        copy_to_clipboard_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        copy_to_clipboard_button.bind("<ButtonPress>", lambda event : copy_to_clipboard(text_results))

        text = ScrolledText(results_frame)
        text.pack(side="left", ipady=150)
        text.bind("<KeyPress>", lambda x:x)
        # text.insert("end", "Hello"+"\n"*40+"World.")
        text.insert("end", text_results)

        results_frame.pack()


        # table_frame = tk.Frame(results_window)


    def cube_inputs(self, x, return_func, event):
        x = self.process_inputs(x,event)
        return_func([xi**3 for xi in x])
    
    def randomise_inputs(self, return_func, event):
        random_integers = [np.random.randint(1,10) for i in range(16)]
        random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
        return_func(random_rationals)

    def process_inputs(self, x, event):
        x = [self.convert_input_to_rational(xi) for xi in x]
        return x



    def process_and_display_inputs(self, x, event):
        
        x = self.process_inputs(x, event)
        
        trace_length_reduction_interface = TraceLengthReductionInterface(XCoords(x))

        trace_reduction_results = trace_length_reduction_interface.trace_reduction()
        length_reduction_results = trace_length_reduction_interface.length_reduction()
        
        self.display_results(trace_reduction_results, title=f"Trace Reduction Results (X-coords: {tuple(x)})")
        self.display_results(length_reduction_results, title=f"Length Reduction Results (X-coords: {tuple(x)})")
        
        

    def convert_input_to_rational(self, input):
        if '/' in input:
            input = input.rsplit('/')
            return sp.Number(input[0])/sp.Number(input[1])
        return sp.Number(input)
    
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