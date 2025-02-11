import numpy as np
import matplotlib.pyplot as plt


def dot_product(f: "first function", g: "second function", a: "lower limit", b: "upper limit"):
    '''
        Function to calculate dot product of two functions

        f, g - functions
        a, b - limits

        result: dot product of f and g on [a, b]
        
    '''

    x = np.linspace(a, b, 10000)
    dx = x[1] - x[0]
    return np.dot(f(x), g(x)) * dx


def get_sincosmt(T: "period"):
    '''
        Function to get sin(n * omega * t) and cos(n * omega * t) functions

        T - period

        result: sin(2 * pi * n / T * t), cos(2 * pi * n / T * t)
    '''

    sinmt = lambda n, t: np.sin(2 * np.pi * n / T * t)
    cosmt = lambda n, t: np.cos(2 * np.pi * n / T * t)

    return sinmt, cosmt


def get_expmt(T: "period"):
    '''
        Function to get e^(n * omega * t) function

        T - period

        result: e^(i * 2 * pi * n / T * t)
    '''

    return lambda n, t: np.e ** (1j * 2 * np.pi * n / T * t)


# calculate fourier coefficients
def fourier_coefficients(func: "source function", t0: "lower limit", T: "period", N: "number of coefficients"):
    '''
        Function to calculate fourier coefficients of a function

        func - source function
        t0 - lower limit
        T - period
        N - number of coefficients

        result: a, b - fourier coefficients lists 
    '''
    sinmt, cosmt = get_sincosmt(T)
    a = []
    b = []
    for n in range(N + 1):
        sin = lambda t: sinmt(n, t)
        cos = lambda t: cosmt(n, t)

        a.append(dot_product(func, cos, t0, t0 + T) * 2 / T)
        b.append(dot_product(func, sin, t0, t0 + T) * 2 / T)
    return a, b


def fourier_exp_coefficients(func: "source function", t0: "lower limit", T: "period", N: "number of coefficients"):
    ''' 
        Function to calculate exponential fourier coefficients of a function

        func - source function
        t0 - lower limit
        T - period
        N - number of coefficients

        result: c - exponential fourier coefficients list
    '''

    expmt = get_expmt(T)
    c = []
    for n in range(-N, N + 1):
        exp = lambda t: expmt(-n, t)
        c.append(dot_product(func, exp, t0, t0 + T) / T)

    return c


def print_fourier_coefficients(a, b):
    ''' 
        Function to print fourier coefficients

        a, b - fourier coefficients
    '''

    for i in range(len(a)):
        print(f'a_{i} = \t{a[i]:.5f}, \tb_{i} = \t{b[i]:.5f}')


def print_fourier_exp_coefficients(c):
    ''' 
        Function to print exponential fourier coefficients

        c - exponential fourier coefficients
    '''

    for i in range(len(c)):
        print(f'c_{i - len(c) // 2} = \t{c[i]:.5f}')



def fourier_func(a, b, T: "period"):
    ''' 
        Function to get fourier function from fourier coefficients

        a, b - fourier coefficients
        T - period

        result: fourier function f(t)
    '''

    sinmt, cosmt = get_sincosmt(T)
    return lambda t: a[0] / 2 + sum([a[n] * cosmt(n, t) + b[n] * sinmt(n, t) for n in range(1, len(a))])


def fourier_exp_func(c, T: "period"):
    '''
        Function to get fourier function from exponential fourier coefficients

        c - exponential fourier coefficients
        T - period

        result: fourier function f(t)
    '''

    expmt = get_expmt(T)
    return lambda t: sum([c[n + len(c) // 2] * expmt(n, t) for n in range(-(len(c) // 2), len(c) // 2)])


def fourierise(func: "source function", t0: "lower limit", T: "period", N: "number of coefficients"):
    ''' 
        Function to get fourier function from source function

        func - source function
        t0 - lower limit
        T - period
        N - number of coefficients

        result: fourier partial sum function f(t)
    '''

    a, b = fourier_coefficients(func, t0, T, N)
    return fourier_func(a, b, T)


def fourierise_exp(func: "source function", t0: "lower limit", T: "period", N: "number of coefficients"):
    ''' 
        Function to get fourier exponential function from source function

        func - source function
        t0 - lower limit
        T - period
        N - number of coefficients

        result: fourier exponential partial sum function f(t)
    '''

    c = fourier_exp_coefficients(func, t0, T, N)
    return fourier_exp_func(c, T)


def plot_func(func: "source function", fourier_func: "partial sum function", t0: "lower limit", T: "period", caption: "figure caption" = '', title: "file name" = ''):
    '''
        Function to plot source and fourier functions

        func - source function
        fourier_func - fourier partial sum function
        t0 - lower limit
        T - period
        caption - figure caption
        title - file name to save

        result: plot with lower and upper limits t0 and t0 + T, plot
        of source and fourier functions if fourier_func != func 
        else plot of source function only 

        left and right limits are calculated as t0 - T * 0.5 and t0 + 1.5 * T
    '''

    # limits
    x_min = t0 - T * 0.5
    x_max = t0 + 1.5 * T
    ymin = min(func(np.linspace(x_min, x_max, 100))) 
    ymax = max(func(np.linspace(x_min, x_max, 100))) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    x = np.linspace(x_min, x_max, 1000)
    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(x, func(x))
    if fourier_func != func:
        plt.plot(x, fourier_func(x))
        plt.plot([t0, t0], [ymin, ymax], 'g--')
        plt.plot([t0 + T, t0 + T], [ymin, ymax], 'g--')
    # add labels
    plt.xlabel('t')
    plt.ylabel('f(t)')
    if fourier_func != func:
        plt.legend(['Source function', 'Fourier function'], loc='upper right')
    else:
        plt.legend(['Source function'], loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def plot_parametric_func(func: "source function", fourier_func: "fourier function", t0: "lower bound", T: "period", caption: "figure caption" = '', title: "file name" = ''):
    '''
        Function to plot parametric source and fourier functions

        func - source function
        fourier_func - fourier partial sum function
        t0 - lower limit
        T - period
        caption - figure caption
        title - file name to save

        result: plot of source and fourier functions if fourier_func != func 
        else plot of source function only 
    '''

    t = np.linspace(t0, t0 + T, 1000)
    plt.figure(figsize=(8, 5))
    plt.plot(func(t).real, func(t).imag)
    if fourier_func != func:
        plt.plot(fourier_func(t).real, fourier_func(t).imag)
    # add labels
    plt.xlabel('Re')
    plt.ylabel('Im')
    if fourier_func != func:
        plt.legend(['Source function', 'Fourier function'], loc='upper right')
    else:
        plt.legend(['Source function'], loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png') 


def calc_and_plot(func: "source function", t0: "lower limit", T: "period", N: "number of coefficients", title: "file name"='untitled'):
    ''' 
        Function to calculate and plot source and fourier functions

        func - source function
        t0 - lower limit
        T - period
        N - list of number of coefficients
        title - file name to save
    ''' 

    plot_func(func, func, t0, T, 'Source function', title)
    for n in N: # plot for each N
        right_caption = f'Fourier function, N = {n}'
        fourier_func = fourierise(func, t0, T, n)
        
        plot_func(func, fourier_func, t0, T, right_caption, title + f'_N_{n}')


def calc_and_plot_exp(func: "source function", t0: "lower limit", T: "period", N: "number of coefficients", title: "file name"='untitled'):   
    '''
        Function to calculate and plot source and fourier exponential functions

        func - source function
        t0 - lower limit
        T - period
        N - list of number of coefficients
        title - file name to save
    '''
    
    plot_func(func, func, t0, T, 'Source function', title)
    for n in N:
        right_caption = f'Fourier function (exp), N = {n}'
        fourier_func = fourierise_exp(func, t0, T, n)
        
        plot_func(func, fourier_func, t0, T, right_caption, title + f'_N_{n}')


def calc_and_plot_parametric(func: "source function", t0: "lower limit", T: "period", N: "number of coefficients", title: "file name"='untitled'):
    '''
        Function to calculate and plot parametric source and fourier functions

        func - source function
        t0 - lower limit
        T - period
        N - list of number of coefficients
        title - file name to save
    '''

    plot_parametric_func(func, func, t0, T, 'Source function', title)
    for n in N:
        right_caption = f'Fourier function, N = {n}'
        fourier_func = fourierise_exp(func, t0, T, n)
        
        plot_parametric_func(func, fourier_func, t0, T, right_caption, title + f'_N_{n}')


def perseval_check(func: "source function", N: "number of coefficients for check"):
    '''
        Function to check Parseval's identity

        func - source function
        N - number of coefficients for check

        result: difference between norm squared and sum of |c_i|^2, 
        difference between norm squared and sum of |a_i|^2 + |b_i|^2
    '''

    abs_func = np.vectorize(lambda x: abs(func(x)))
    norm_squared = dot_product(abs_func, abs_func, -np.pi, np.pi)
    a, b = fourier_coefficients(abs_func, -np.pi, 2 * np.pi, N)
    c = fourier_exp_coefficients(abs_func, -np.pi, 2 * np.pi, N)
    c_sum = 2 * np.pi * sum(abs(c[i]) ** 2 for i in range(len(c)))
    ab_sum = np.pi * (a[0] ** 2 / 2 + sum([a[i] ** 2 + b[i] ** 2 for i in range(1, N + 1)]))
    print(f'Norm squared: {norm_squared:.5f}, \tSum of |c_i|^2: {c_sum:.5f}, \tSum of |a_i|^2 + |b_i|^2: {ab_sum:.5f}')
    return abs(norm_squared - c_sum), abs(norm_squared - ab_sum)


R = 3
T = 4

def func5(t):
    t = (t + T/8) % T - T/8
    real = -1
    if -T/8 <= t < T/8:
        real = R
    if T/8 <= t < 3 * T / 8:
        real = 2 * R - 8 * R * t / T 
    if 3 * T / 8 <= t < 5 * T / 8:
        real = -R
    if 5 * T / 8 <= t <= 7 * T / 8:
        real = -6 * R + 8 * R * t / T

    imag = -1
    if -T/8 <= t < T/8:
        imag = 8 * R * t / T
    if T/8 <= t < 3 * T / 8:
        imag = R
    if 3 * T / 8 <= t < 5 * T / 8:
        imag = 4 * R - 8 * R * t / T
    if 5 * T / 8 <= t <= 7 * T / 8:
        imag = -R

    return real + 1j * imag
    

# Example 1
# func = np.vectorize(lambda x: 1 if 0 <= (x - 1) % 3 < 1 else 2)
# print("Func 1")
# a, b = fourier_coefficients(func, 1, 3, 3)
# print_fourier_coefficients(a, b)
# c = fourier_exp_coefficients(func, 1, 3, 3)
# print_fourier_exp_coefficients(c)
# perseval_check(func, 300)
# calc_and_plot(func, 1, 3, [1, 2, 5, 15, 30], './media/plots/func_1')
# calc_and_plot_exp(func, 1, 3, [1, 2, 5, 15, 30], './media/plots/func_1_exp')

# # Example 2
# func = np.vectorize(lambda x: np.sin(5/2 * np.cos(x)))
# print("Func 2")
# a, b = fourier_coefficients(func, -np.pi, 2*np.pi, 3)
# print_fourier_coefficients(a, b)
# c = fourier_exp_coefficients(func, -np.pi, 2*np.pi, 3)
# print_fourier_exp_coefficients(c)
# perseval_check(func, 300)
# calc_and_plot(func, -np.pi, 2 * np.pi, [1, 2, 3, 4, 5], './media/plots/func')
# calc_and_plot_exp(func, -np.pi, 2 * np.pi, [1, 2, 3, 4, 5], './media/plots/func')

# # Example 3
# func = np.vectorize(lambda x: abs(np.cos(2 * x) * np.sin(x)))
# print("Func 3")
# a, b = fourier_coefficients(func, 0, np.pi, 3)
# print_fourier_coefficients(a, b)
# c = fourier_exp_coefficients(func, 0, np.pi, 3)
# print_fourier_exp_coefficients(c)
# perseval_check(func, 300)
# calc_and_plot(func, 0, np.pi, [1, 2, 3, 4, 5], './media/plots/func_3')
# calc_and_plot_exp(func, 0, np.pi, [1, 2, 3, 4, 5], './media/plots/func_3_exp')

# # Example 4
# func = np.vectorize(lambda x: np.sin(x) ** 3 - np.cos(x))
# print("Func 4")
# a, b = fourier_coefficients(func, 0, 2 * np.pi, 3)
# print_fourier_coefficients(a, b)
# c = fourier_exp_coefficients(func, 0, 2 * np.pi, 3)
# print_fourier_exp_coefficients(c)
# perseval_check(func, 300)
# calc_and_plot(func, 0, 2 * np.pi, [1, 2, 3, 4, 5], './media/plots/func_4') 
# calc_and_plot_exp(func, 0, 2 * np.pi, [1, 2, 3, 4, 5], './media/plots/func_4_exp')

# # Example 5
# func = np.vectorize(func5)
# perseval_check(func, 300)
# c = fourier_exp_coefficients(func, -T/8, T, 3)
# print_fourier_exp_coefficients(c)
# calc_and_plot_parametric(func, -T/8, T, [1, 2, 3, 5, 10], './media/plots/func_5')

# plot_func(lambda x: func(x).real, lambda x: func(x).real, -T/8, T, 'Source function (Re)', f'./media/plots/func_5_real')
# plot_func(lambda x: func(x).imag, lambda x: func(x).imag, -T/8, T, 'Source function (Im)', f'./media/plots/func_5_imag')

# for n in [1, 2, 3, 5, 10]:
#     fourier_func = fourierise_exp(func, -T/8, T, n)
#     plot_func(lambda x: func(x).real, lambda x: fourier_func(x).real, -T/8, T, f'Fourier function (Re), N = {n}', f'./media/plots/func_5_real_N_{n}')
#     plot_func(lambda x: func(x).imag, lambda x: fourier_func(x).imag, -T/8, T, f'Fourier function (Im), N = {n}', f'./media/plots/func_5_imag_N_{n}')


func = np.vectorize(lambda x: 1 if 0 <= (x - np.pi) % (2*np.pi) < np.pi else 3)
print("Func 1")
a, b = fourier_coefficients(func, -np.pi, 2*np.pi, 3)
print_fourier_coefficients(a, b)
c = fourier_exp_coefficients(func, -np.pi, 2*np.pi, 3)
print_fourier_exp_coefficients(c)
# perseval_check(func, 300)
calc_and_plot(func, -np.pi, 2*np.pi, [1, 2, 5, 15, 30], './media/plots/test')
calc_and_plot_exp(func, -np.pi, 2*np.pi, [1, 2, 5, 15, 30], './media/plots/test')

plt.show()
