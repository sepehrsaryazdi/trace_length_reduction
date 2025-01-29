from trace_length_reduction.reduction import ReductionResults

class LengthTracePlot():
    """
    Class for initialising and visualising the length trace plot.
    """
    def __init__(self, trace_reduction_results, length_reduction_results):
        assert isinstance(trace_reduction_results, ReductionResults), f"Error: {trace_reduction_results} must be an instance of ReductionResults."
        assert isinstance(length_reduction_results, ReductionResults), f"Error: {length_reduction_results} must be an instance of ReductionResults."

        print(trace_reduction_results)