import numpy as np
import sympy as sp


def generate_cusp_example():
    random_integers = [np.random.randint(1,10) for i in range(16)]
    random_integers = [1,1]*2 +[1, 2]*6 # cusp shorter
    random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
    random_rationals[2]=sp.Number(1)/random_rationals[3]
    random_rationals[4]=sp.Number(1)/random_rationals[5]
    random_rationals[6]=sp.Number(1)/random_rationals[7]
    random_rationals[1] = sp.Number(1)/(random_rationals[0])
    return random_rationals

def generate_shorter_hyperbolic_example():
    random_integers = [np.random.randint(1,10) for i in range(16)]
    random_integers = [1,1]*2 +[5, 10]*6 # hyperbolic end shorter
    # random_integers = [1,1]*2 +[10, 10]*6 # hyperbolic end longer
    random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
    random_rationals[2]=sp.Number(1)/random_rationals[3]
    random_rationals[4]=sp.Number(2)/random_rationals[5]
    random_rationals[6]=sp.Number(1)/random_rationals[7]
    random_rationals[1] = sp.Number(1)/(sp.Number(2)*random_rationals[0])
    return random_rationals

def generate_longer_hyperbolic_example():
    random_integers = [np.random.randint(1,10) for i in range(16)]
    # random_integers = [1,1]*2 +[5, 10]*6 # hyperbolic end shorter
    random_integers = [1,1]*2 +[10, 10]*6 # hyperbolic end longer
    random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
    random_rationals[2]=sp.Number(1)/random_rationals[3]
    random_rationals[4]=sp.Number(2)/random_rationals[5]
    random_rationals[6]=sp.Number(1)/random_rationals[7]
    random_rationals[1] = sp.Number(1)/(sp.Number(2)*random_rationals[0])
    return random_rationals


def generate_shorter_special_example():   
    random_integers = [np.random.randint(1,10) for i in range(16)]
    random_integers = [1,1]*2 +[20, 10]*6 # special end shorter
    # random_integers = [1,1]*2 +[15, 10]*6 # special end longer
    random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
    random_rationals[2]=sp.Number(1)/random_rationals[3]
    random_rationals[4]=sp.Number(1)/random_rationals[5]
    random_rationals[6]=sp.Number(1)/random_rationals[7]
    random_rationals[1] = sp.Number(1)/(sp.Number(2)*random_rationals[0])
    return random_rationals


def generate_longer_special_example():
    random_integers = [np.random.randint(1,10) for i in range(16)]
    # random_integers = [1,1]*2 +[20, 10]*6 # special end shorter
    random_integers = [1,1]*2 +[15, 10]*6 # special end longer
    random_rationals = [sp.Number(random_integers[2*i])/sp.Number(random_integers[2*i+1]) for i in range(8)]
    random_rationals[2]=sp.Number(1)/random_rationals[3]
    random_rationals[4]=sp.Number(1)/random_rationals[5]
    random_rationals[6]=sp.Number(1)/random_rationals[7]
    random_rationals[1] = sp.Number(1)/(sp.Number(2)*random_rationals[0])
    return random_rationals


def give_all_examples():
    return [generate_cusp_example, generate_shorter_hyperbolic_example, generate_longer_hyperbolic_example, generate_shorter_hyperbolic_example, generate_longer_special_example]

