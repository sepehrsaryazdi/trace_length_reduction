import numpy as np
import sympy as sp


def generate_arithmetic_torus_example():
    return [sp.Number(1)]*8


def generate_hyperbolic_surface_shorter_end_example():
    return [sp.Number(1), sp.Number(1)] + [sp.Pow(sp.Number(6)/sp.Number(5),1/sp.Number(3))]*6


def generate_hyperbolic_surface_longer_end_example():
    return [sp.Number(1), sp.Number(1)] + [sp.Pow(sp.Number(3)/sp.Number(2),1/sp.Number(3))]*6


def generate_cusp_end_example():
    random_integers = [1,1]*2 +[1, 2]*6 # cusp shorter
    random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
    random_rationals[2]=sp.Number(1)/random_rationals[3]
    random_rationals[4]=sp.Number(1)/random_rationals[5]
    random_rationals[6]=sp.Number(1)/random_rationals[7]
    random_rationals[1] = sp.Number(1)/(random_rationals[0])
    return random_rationals

def generate_shorter_hyperbolic_end_example():
    random_integers = [1,1]*2 +[5, 10]*6 # hyperbolic end shorter
    # random_integers = [1,1]*2 +[10, 10]*6 # hyperbolic end longer
    random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
    random_rationals[2]=sp.Number(1)/random_rationals[3]
    random_rationals[4]=sp.Number(2)/random_rationals[5]
    random_rationals[6]=sp.Number(1)/random_rationals[7]
    random_rationals[1] = sp.Number(1)/(sp.Number(2)*random_rationals[0])
    return random_rationals

def generate_longer_hyperbolic_end_example():
    # random_integers = [1,1]*2 +[5, 10]*6 # hyperbolic end shorter
    random_integers = [1,1]*2 +[10, 10]*6 # hyperbolic end longer
    random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
    random_rationals[2]=sp.Number(1)/random_rationals[3]
    random_rationals[4]=sp.Number(2)/random_rationals[5]
    random_rationals[6]=sp.Number(1)/random_rationals[7]
    random_rationals[1] = sp.Number(1)/(sp.Number(2)*random_rationals[0])
    return random_rationals


def generate_shorter_special_end_example():   
    random_integers = [1,1]*2 +[20, 10]*6 # special end shorter
    # random_integers = [1,1]*2 +[15, 10]*6 # special end longer
    random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
    random_rationals[2]=sp.Number(1)/random_rationals[3]
    random_rationals[4]=sp.Number(1)/random_rationals[5]
    random_rationals[6]=sp.Number(1)/random_rationals[7]
    random_rationals[1] = sp.Number(1)/(sp.Number(2)*random_rationals[0])
    return random_rationals


def generate_longer_special_end_example():
    # random_integers = [1,1]*2 +[20, 10]*6 # special end shorter
    random_integers = [1,1]*2 +[15, 10]*6 # special end longer
    random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
    random_rationals[2]=sp.Number(1)/random_rationals[3]
    random_rationals[4]=sp.Number(1)/random_rationals[5]
    random_rationals[6]=sp.Number(1)/random_rationals[7]
    random_rationals[1] = sp.Number(1)/(sp.Number(2)*random_rationals[0])
    return random_rationals






def give_all_examples():
    return [generate_arithmetic_torus_example, generate_hyperbolic_surface_shorter_end_example, generate_hyperbolic_surface_longer_end_example, generate_cusp_end_example, generate_shorter_hyperbolic_end_example, generate_longer_hyperbolic_end_example, generate_shorter_hyperbolic_end_example, generate_longer_special_end_example]
    # return [generate_hyperbolic_surface_shorter_end_example,generate_hyperbolic_surface_longer_end_example]