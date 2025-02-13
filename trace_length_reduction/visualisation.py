from trace_length_reduction.reduction import ReductionResults
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import tkinter as tk
from tkinter import ttk


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


class MenuApp(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        # self.pack(anchor='nw')


class Menu:
    """
    Class for visualising menu for seeing examples or minimising a set of X-coordinates.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Trace Length Reduction Algorithm Menu")
        self.root.geometry("520x120")
        self.menubar = tk.Menu(self.root)        
        self.app = MenuApp(self.root)
        self.button_frame = ttk.Frame(self.root)    
        self.button_frame.pack(side="top", pady=(20,0))

        self.examples_button= ttk.Button(self.button_frame, text="Examples")
        self.examples_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        self.examples_button.bind("<ButtonPress>", lambda x:x)


        self.minimise_button= ttk.Button(self.button_frame, text="Minimise X-coordinates")
        self.minimise_button.pack(side="left", padx=25, ipadx=20, ipady=20)
        self.minimise_button.bind("<ButtonPress>", lambda x:x)



    def show(self):
        self.app.mainloop()