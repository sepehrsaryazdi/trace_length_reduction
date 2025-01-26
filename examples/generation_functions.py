import numpy as np
import sympy as sp


def generate_arithmetic_torus_shorter_cusp_example():
    return [sp.Number(1)]*8


def generate_hyperbolic_surface_shorter_hyperbolic_end_example():
    return [sp.Number(1), sp.Number(1)] + [sp.Pow(sp.Number(6)/sp.Number(5),1/sp.Number(3))]*6


def generate_hyperbolic_surface_longer_hyperbolic_end_example():
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


def generate_inbetween_hyperbolic_end_example():
    r=5
    s=5 # s=5
    random_integers = [1,1]*2 +[1,1] + [s*5+2*r,s*10+3*r] + [1,1] + [s*5+2*r, s*10+3*r] + [1,1] + [s*5+2*r, s*10+3*r] # hyperbolic end shorter
    # random_integers = [1,1]*2 +[10, 10]*6 # hyperbolic end longer
    random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
    random_rationals[2]=sp.Number(1)/random_rationals[3]
    random_rationals[4]=sp.Number(2)/random_rationals[5]
    random_rationals[6]=sp.Number(1)/random_rationals[7]
    random_rationals[1] = sp.Number(2+3)/(sp.Number(2+10)*random_rationals[0])
    return random_rationals


def random_rationals():
    # random_integers = [np.random.randint(1,10) for i in range(16)]
    # random_integers = [1,216, 125,27, 27,125, 27,8, 1,216, 512,1, 125,1, 1,8]
    
    random_integers = [27,64, 64,125, 1,27, 1,8, 64,1, 8,729, 512,1, 343,216]
    random_integers = [sp.Pow(random_integers[i],1/sp.Number(3)) for i in range(len(random_integers))]
    random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
    return random_rationals





def give_all_examples():
    return [generate_arithmetic_torus_shorter_cusp_example, generate_hyperbolic_surface_shorter_hyperbolic_end_example, generate_hyperbolic_surface_longer_hyperbolic_end_example, generate_cusp_end_example, generate_shorter_hyperbolic_end_example, generate_longer_hyperbolic_end_example, generate_shorter_special_end_example, generate_longer_special_end_example, generate_inbetween_hyperbolic_end_example]
    # return [random_rationals]
    # return [generate_arithmetic_torus_example, generate_hyperbolic_surface_shorter_end_example, generate_hyperbolic_surface_longer_end_example]