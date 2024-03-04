import numpy as np
import matplotlib.pyplot as plt


# calculate dot product of two functions 
def dot_product(f, g, a, b):
    x = np.linspace(a, b, 10000)
    dx = x[1] - x[0]
    return np.dot(f(x), g(x)) * dx


# get sin(omega * t) and cos(omega * t) functions
def get_sincosmt(T):
    sinmt = lambda x: np.sin(2 * np.pi * x[0] / T * x[1])
    cosmt = lambda x: np.cos(2 * np.pi * x[0] / T * x[1])

    return sinmt, cosmt


# git exp(-i * omega * t) 
def get_expmt(T):
    return lambda x: np.e ** (1j * 2 * np.pi * x[0] / T * x[1])


# calculate fourier coefficients
def fourier(func, t0, T, N):
    sinmt, cosmt = get_sincosmt(T)
    a = []
    b = []
    for i in range(N + 1):
        sin = lambda x: sinmt([i, x])
        cos = lambda x: cosmt([i, x])

        a.append(dot_product(func, cos, t0, t0 + T) * 2 / T)
        b.append(dot_product(func, sin, t0, t0 + T) * 2 / T)
    return a, b


def fourier_exp(func, t0, T, N):
    expmt = get_expmt(T)
    c = []
    for i in range(-N, N + 1):
        exp = lambda x: expmt([-i, x])
        c.append(dot_product(func, exp, t0, t0 + T) / T)

    return c

def print_fourier_coefficients(a, b):
    for i in range(len(a)):
        print(f'a_{i} = \t{a[i]:.5f}, \tb_{i} = \t{b[i]:.5f}')


def print_fourier_exp_coefficients(c):
    for i in range(len(c)):
        print(f'c_{i - len(c) // 2} = \t{c[i]:.5f}')


def fourier_func(a, b, T, N):
    sinmt, cosmt = get_sincosmt(T)
    return lambda x: a[0] / 2 + sum([a[i] * cosmt([i, x]) + b[i] * sinmt([i, x]) for i in range(1, N + 1)])


def fourier_exp_func(c, T, N):
    expmt = get_expmt(T)
    return lambda x: sum([c[i + len(c) // 2] * expmt([i, x]) for i in range(-N, N + 1)])


def fourierise(func, t0, T, N):
    a, b = fourier(func, t0, T, N)
    return fourier_func(a, b, T, N)


def fourierise_exp(func, t0, T, N):
    c = fourier_exp(func, t0, T, N)
    return fourier_exp_func(c, T, N)


def plot_func(func, fourier_func, t0, T, caption = '', title = ''):
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
    plt.savefig(title + '.png')


def plot_parametric_func(func, fourier_func, t0, T, caption = '', title = ''):
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
    plt.savefig(title + '.png') 


def calc_and_plot(func, t0, T, N, title):
    plot_func(func, func, t0, T, 'Source function', title)
    for n in N:
        right_caption = f'Fourier function, N = {n}'
        fourier_func = fourierise(func, t0, T, n)
        
        plot_func(func, fourier_func, t0, T, right_caption, title + f'_N_{n}')


def calc_and_plot_exp(func, t0, T, N, title):   
    plot_func(func, func, t0, T, 'Source function', title)
    for n in N:
        right_caption = f'Fourier function (exp), N = {n}'
        fourier_func = fourierise_exp(func, t0, T, n)
        
        plot_func(func, fourier_func, t0, T, right_caption, title + f'_N_{n}')


def calc_and_plot_parametric(func, t0, T, N, title):
    plot_parametric_func(func, func, t0, T, 'Source function', title)
    for n in N:
        right_caption = f'Fourier function, N = {n}'
        fourier_func = fourierise_exp(func, t0, T, n)
        
        plot_parametric_func(func, fourier_func, t0, T, right_caption, title + f'_N_{n}')


def perseval_check(func, N):
    norm_squared = dot_product(func, func, -np.pi, np.pi)
    a, b = fourier(func, -np.pi, 2 * np.pi, N)
    c = fourier_exp(func, -np.pi, 2 * np.pi, N)
    c_sum = 2 * np.pi * sum(abs(c[i]) ** 2 for i in range(len(c)))
    ab_sum = np.pi * (a[0] ** 2 / 2 + sum([a[i] ** 2 + b[i] ** 2 for i in range(1, N + 1)]))
    print(f'Norm squared: {norm_squared:.5f}, \tSum of |c_i|^2: {c_sum:.5f}, \tSum of |a_i|^2 + |b_i|^2: {ab_sum:.5f}')


def perseval_check_exp(func, N, t0, T):
    f = np.vectorize(lambda x: abs(func(x)))
    norm_squared = dot_product(f, f, -np.pi, np.pi)
    c = fourier_exp(func, -np.pi, 2 * np.pi, N)
    c_sum = 2 * np.pi * sum(abs(c[i]) ** 2 for i in range(len(c)))
    print(f'Norm squared: {norm_squared:.5f}, \tSum of |c_i|^2: {c_sum:.5f}')


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
func = np.vectorize(lambda x: 1 if 0 <= (x - 1) % 3 < 1 else 2)
print("Func 1")
# a, b = fourier(func, 1, 3, 3)
# print_fourier_coefficients(a, b)
# c = fourier_exp(func, 1, 3, 3)
# print_fourier_exp_coefficients(c)
# perseval_check(func, 300)
# calc_and_plot(func, 1, 3, [1, 2, 5, 15, 30], './media/plots/func_1')
# calc_and_plot_exp(func, 1, 3, [1, 2, 5, 15, 30], './media/plots/func_1_exp')
# calc_and_plot(func, -np.pi, 2 * np.pi, [1, 2, 5, 15, 30], './media/plots/func')
# calc_and_plot_exp(func, -np.pi, 2 * np.pi, [1, 2, 5, 15, 30], './media/plots/func')

# Example 2
func = np.vectorize(lambda x: np.sin(5/2 * np.cos(x)))
print("Func 2")
# a, b = fourier(func, -np.pi, 2*np.pi, 3)
# print_fourier_coefficients(a, b)
# c = fourier_exp(func, -np.pi, 2*np.pi, 3)
# print_fourier_exp_coefficients(c)
# perseval_check(func, 300)
# calc_and_plot(func, -np.pi, 2 * np.pi, [1, 2, 3, 4, 5], './media/plots/func')
# calc_and_plot_exp(func, -np.pi, 2 * np.pi, [1, 2, 3, 4, 5], './media/plots/func')

# # Example 3
func = np.vectorize(lambda x: abs(np.cos(2 * x) * np.sin(x)))
print("Func 3")
# a, b = fourier(func, 0, np.pi, 3)
# print_fourier_coefficients(a, b)
# c = fourier_exp(func, 0, np.pi, 3)
# print_fourier_exp_coefficients(c)
# perseval_check(func, 300)
# calc_and_plot(func, 0, np.pi, [1, 2, 3, 4, 5], './media/plots/func_3')
# calc_and_plot_exp(func, 0, np.pi, [1, 2, 3, 4, 5], './media/plots/func_3_exp')
# calc_and_plot(func, -np.pi, 2 * np.pi, [1, 2, 3, 4, 5], './media/plots/func')
# calc_and_plot_exp(func, -np.pi, 2 * np.pi, [1, 2, 3, 4, 5], './media/plots/func')


# # Example 4
func = np.vectorize(lambda x: np.sin(x) ** 3 - np.cos(x))
print("Func 4")
# a, b = fourier(func, 0, 2 * np.pi, 3)
# print_fourier_coefficients(a, b)
# c = fourier_exp(func, 0, 2 * np.pi, 3)
# print_fourier_exp_coefficients(c)
# perseval_check(func, 300)
# calc_and_plot(func, 0, 2 * np.pi, [1, 2, 3, 4, 5], './media/plots/func_4') 
# calc_and_plot_exp(func, 0, 2 * np.pi, [1, 2, 3, 4, 5], './media/plots/func_4_exp')
# calc_and_plot(func, -np.pi, 2 * np.pi, [1, 2, 3, 4, 5], './media/plots/func') 
# calc_and_plot_exp(func, -np.pi, 2 * np.pi, [1, 2, 3, 4, 5], './media/plots/func')


# Example 5
func = np.vectorize(func5)
perseval_check_exp(func, 300, -T/8, T)
# c = fourier_exp(func, -T/8, T, 3)
# print_fourier_exp_coefficients(c)
calc_and_plot_parametric(func, -T/8, T, [1, 2, 3, 5, 10], './media/plots/func_5')

# plot_func(lambda x: func(x).real, lambda x: func(x).real, -T/8, T, 'Source function (Re)', f'./media/plots/func_5_real')
# plot_func(lambda x: func(x).imag, lambda x: func(x).imag, -T/8, T, 'Source function (Im)', f'./media/plots/func_5_imag')

# for n in [1, 2, 3, 5, 10]:
#     fourier_func = fourierise_exp(func, -T/8, T, n)
#     plot_func(lambda x: func(x).real, lambda x: fourier_func(x).real, -T/8, T, f'Fourier function (Re), N = {n}', f'./media/plots/func_5_real_N_{n}')
#     plot_func(lambda x: func(x).imag, lambda x: fourier_func(x).imag, -T/8, T, f'Fourier function (Im), N = {n}', f'./media/plots/func_5_imag_N_{n}')


plt.show()

