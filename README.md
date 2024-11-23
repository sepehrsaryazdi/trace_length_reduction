# Trace Length Reduction Algorithms

The extended algorithms for length and trace reduction on convex projective surfaces are presented in this repository. Their implementations are done symbolically through the [SymPy](https://www.sympy.org/en/index.html) library. The only potential technical issue lies in the calculation of the eigenvalues for 3 x 3 matrices, which relies on the cubic root formula in the SymPy implementation. This means that the eigenvalues will be rounded to a certain precision, although this can be set arbitrarily high in the implementation.

To install, run the following:

```
pip3 install numpy sympy matplotlib 
```

To visualise the eigenvalue plots with LaTeX rendered labels, a LaTeX distribution is expected to be installed on the host machine for these lines to correctly execute in the `plot_eigenvalues_and_traces` function. 

```
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
```

If a LaTeX distribution is not available, one can comment out the `plot_eigenvalues_and_traces` function call to run the algorithm nonetheless.

To run, do the following:

1. Either input custom random integers into the 16 x 1 vector `random_integers`, or comment it out to allow the random number generator to produce numbers in the range 1 to 10 from the `np.random.randint(1,10)` function call. By default, integers corresponding to an infinite volume torus will override the randomly generated integers.

After this step, the integers will be divided in the respective order to form 8 rationals, i.e. if a precedes b, then the rational will be a/b.

To obtain the X-coordinates, the rationals will be cubed and their cube roots will be implicitly kept as the original rationals. These are then fed into the `compute_translation_matrix_torus` function that returns the holonomy representations A, B of the canonical generators of the once-punctured torus. 

2. Run the following command:

```
python3 trace_and_length_minimisation_x_coordinates.py
```

After running, the canonical generators A, B will be printed out in LaTeX code. Note that these are not necessarily ordered by length or trace.
