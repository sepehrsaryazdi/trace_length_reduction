import sympy as sp
from trace_length_reduction.reduction import XCoords, TraceLengthReductionInterface
from trace_length_reduction.visualisation import LengthTracePlot, Menu

random_integers = [1,1]*2 +[15, 10]*6 # special end longer
random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
random_rationals[2]=sp.Number(1)/random_rationals[3]
random_rationals[4]=sp.Number(1)/random_rationals[5]
random_rationals[6]=sp.Number(1)/random_rationals[7]
random_rationals[1] = sp.Number(1)/(sp.Number(2)*random_rationals[0])

# random_integers = [1,216, 125,27, 27,125, 27,8, 1,216, 512,1, 125,1, 1,8]
random_integers = [27,64, 64,125, 1,27, 1,8, 64,1, 8,729, 512,1, 343,216]
random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]

trace_length_reduction_interface = TraceLengthReductionInterface(XCoords(random_rationals))

trace_reduction_results = trace_length_reduction_interface.trace_reduction()
length_reduction_results = trace_length_reduction_interface.length_reduction()

# print(trace_reduction_results)
# print(length_reduction_results)

# length_trace_plot = LengthTracePlot(trace_reduction_results, length_reduction_results)

menu = Menu()
menu.show()