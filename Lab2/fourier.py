import numpy as np 
import matplotlib.pyplot as plt 


# calculate dot product of two functions 
def dot_product(f, g, a, b):
    x = np.linspace(a, b, 10000)
    dx = x[1] - x[0]
    return np.dot(f(x), g(x)) * dx


# get a nd b functions 
def fourier(func, T, N, bounds):
    a = lambda omega: (dot_product(func, lambda t: np.cos(omega * t), -bounds, bounds) / np.pi)
    b = lambda omega: (dot_product(func, lambda t: np.sin(omega * t), -bounds, bounds) / np.pi)
    return a, b


a, b = fourier(lambda x: np.sin(x), T = 2 * np.pi, N = 10, bounds = 1000)
print(a(1))