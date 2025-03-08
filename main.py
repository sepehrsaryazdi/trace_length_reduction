import sympy as sp
from trace_length_reduction.reduction import XCoords, TraceLengthReductionInterface
from trace_length_reduction.visualisation import LengthTracePlot, Menu

# example calculation

integers = [27,64, 64,125, 1,27, 1,8, 64,1, 8,729, 512,1, 343,216]
rationals = [sp.Number(integers[2*i])/sp.Number(integers[2*i+1]) for i in range(8)]

trace_length_reduction_interface = TraceLengthReductionInterface(XCoords(rationals))

trace_reduction_results = trace_length_reduction_interface.trace_reduction()
length_reduction_results = trace_length_reduction_interface.length_reduction()

print(trace_reduction_results.get_report())
print(length_reduction_results.get_report())

# interactive menu

menu = Menu()
menu.show()