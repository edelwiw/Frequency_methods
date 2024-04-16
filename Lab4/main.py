import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.signal import tf2zpk, lsim, freqs_zpk
import datetime

def plot_func(X, func, caption, title, legend=['Source function'], labels=['t', 'f(t)']):
    ymin = min(func) 
    ymax = max(func) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(X, func.real)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def plot_freq_response(X, fr, caption, title, legend=['Frequency response'], scale='linear'):
    ymin = min(fr) 
    ymax = max(fr) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.xscale(scale)
    plt.yscale(scale)
    plt.plot(X, fr.real)
    plt.xlabel('\u03C9')
    plt.ylabel('|W(i\u03C9)|')
    if scale == 'linear':
        plt.ylim(ymin, ymax)
        plt.xlim(min(X), max(X))
        difference_array = np.absolute(fr - 1 / np.sqrt(2))
        index = difference_array.argmin()
        plt.plot(np.linspace(min(X), X[index], 100), [1 / np.sqrt(2)] * 100, 'r--', linewidth=1)
        plt.plot([X[index]] * 100, np.linspace(0, fr[index], 100), 'r--', linewidth=1)
    plt.legend(legend + ['1 / \u221A2'], loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def cmp_func(X, funcs, caption, title, legend=['Source function', 'Restored function'], labels=['t', 'f(t)']):
    ymin = min([min(func) for func in funcs])
    ymax = max([max(func) for func in funcs])

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    for f in funcs:
     plt.plot(X, f.real)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def plot_image(X, func, caption, title):
    ymin = min(func.real.min(), func.imag.min())
    ymax = max(func.real.max(), func.imag.max())

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(X, func.real)
    plt.plot(X, func.imag)
    plt.xlabel('\u03C9')
    plt.ylabel('f(\u03C9)')
    plt.legend(['Real', 'Imag'], loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def dot_product(X, f, g):
    dx = X[1] - X[0]
    return np.dot(f, g) * dx


get_fourier_image = lambda X, V, func: np.array([1 / (np.sqrt(2 * np.pi)) * dot_product(X, func, (lambda t: np.e ** (-1j * image_clip * t))(X)) for image_clip in V])
get_fourier_function = lambda X, V, image: np.array([1 / (np.sqrt(2 * np.pi)) * dot_product(V, image, (lambda t: np.e ** (1j * x * t))(V)) for x in X])

get_wave_func = lambda a, t1, t2: np.vectorize(lambda t: a if t1 <= t <= t2 else 0, otypes=[complex])
apply_noise_to_values = lambda X, func, b, c, d: func + b * (np.random.rand(X.size) - 0.5) + c * np.sin(d * X)

differentiate = lambda X, func: np.array([0 if i == 0 else (func[i] - func[i - 1]) / (X[i] - X[i - 1]) for i in range(len(func))])

# clip_outer_image_values = lambda image_clip: lambda X, func: np.array([func[i] if abs(X[i]) <= image_clip else 0 for i in range(len(func))])
# clip_inner_image_values = lambda image_clip: lambda X, func: np.array([func[i] if abs(X[i]) >= image_clip else 0 for i in range(len(func))])
# clip_delta_image_values = lambda pivot, delta: lambda X, func: np.array([0 if (pivot - delta <= X[i] <= pivot + delta  or pivot - delta <= -X[i] <= pivot + delta)else func[i] for i in range(len(func))])

get_first_order_filter = lambda T: tf2zpk([0, 1], [T, 1]) 
get_second_order_filter = lambda T1, T2, T3: tf2zpk([T1 ** 2, 2 * T1, 1], [T2 * T3, T2 + T3, 1])


def first_task(T):
    if not os.path.exists(f'./results/{T}'):
        os.makedirs(f'./results/{T}')

    t = np.linspace(-T, T, T * 100) # array with arguments for functions

    sin = np.sin(t)
    noised_sin = apply_noise_to_values(t, sin, 0.1, 0, 0)

    plot_func(t, noised_sin, caption='Noised function: sin(t)', title=f'./results/{T}/noised_sin', legend=['Noised function'])

    # find derivative of noised sin function
    noised_sin_derivative = differentiate(t, noised_sin)
    plot_func(t, noised_sin_derivative, caption='Derivative of noised function: sin(t)', title=f'./results/{T}/noised_sin_derivative',legend=['Derivative of noised function'])

    noised_sin_image = get_fourier_image(t, t, noised_sin)
    plot_image(t, noised_sin_image, caption='Fourier image of noised function: sin(t)', title=f'./results/{T}/noised_sin_image')

    # derivative image (aka multiply by jw)
    noised_sin_derivative_image = get_fourier_image(t, t, noised_sin) * 1j * t
    plot_image(t, noised_sin_derivative_image, caption='Fourier image of derivative of noised function: sin(t)', title=f'./results/{T}/noised_sin_image_derivative')

    # restore function from image (aka differentiate it)
    restored_noised_sin_derivative = get_fourier_function(t, t, noised_sin_derivative_image)
    plot_func(t, restored_noised_sin_derivative, caption='Restored function from derivative image of noised function: sin(t)', title=f'./results/{T}/noised_sin_image_derivative_restored', legend=['Restored function'])

    # compare restored from image derivative with original derivative and with cos function
    cos = np.cos(t)
    cmp_func(t, funcs=[noised_sin_derivative, restored_noised_sin_derivative, cos], caption='Comparison of sceptral, numerical and true derivate', title=f'./results/{T}/derivative_cmp', legend=['Numerical derivative', 'Spectral derivative', 'cos'])


def filtering(a, b, c, d, t1, t2, L, F, filter, n=0):
    if not os.path.exists(f'./results/second/{n}'):
        os.makedirs(f'./results/second/{n}')
    t_arr = np.linspace(0, L, L * 100)

    # create wave function
    wave_func = get_wave_func(a, t1, t2)(t_arr)
    noised_wave_func = apply_noise_to_values(t_arr, wave_func, b, c, d)

    plot_func(t_arr, wave_func, caption='Wave function', title=f'./results/second/{n}/wave_func', legend=['Source wave function'])
    plot_func(t_arr, noised_wave_func, caption='Noised wave function', title=f'./results/second/{n}/noised_wave_func', legend=['Noised wave function'])

    t_filtered, noised_wave_func_filtered, x_out = lsim(filter, noised_wave_func, t_arr)
    
    plot_func(t_filtered, noised_wave_func_filtered, caption='Wave function after filter', title=f'./results/second/{n}/noised_wave_func_filtered', legend=['Wave function after filter'])
    cmp_func(t_arr, funcs=[wave_func, noised_wave_func_filtered], caption='Comparison of source and filtered wave function', title=f'./results/second/{n}/wave_func_cmp', legend=['Source wave function', 'Filtered wave function'])

    # images of wave function and filtered wave function
    v_arr = np.linspace(-F, F, F * 100)
    wave_func_image = get_fourier_image(t_arr, v_arr, wave_func)
    noised_wave_func_filtered_image = get_fourier_image(t_arr, v_arr, noised_wave_func_filtered)

    plot_image(v_arr, wave_func_image, caption='Fourier image of wave function', title=f'./results/second/{n}/wave_func_image')
    plot_image(v_arr, noised_wave_func_filtered_image, caption='Fourier image of filtered wave function', title=f'./results/second/{n}/noised_wave_func_filtered_image')

    wave_func_image_abs = np.absolute(wave_func_image)
    noised_wave_func_filtered_image_abs = np.absolute(noised_wave_func_filtered_image)

    plot_func(v_arr, wave_func_image_abs, caption='Fourier image of wave function (abs)', title=f'./results/second/{n}/wave_func_image_abs', legend=['Fourier image of wave function (abs)'])
    plot_func(v_arr, noised_wave_func_filtered_image_abs, caption='Fourier image of filtered wave function (abs)', title=f'./results/second/{n}/noised_wave_func_filtered_image_abs', legend=['Fourier image of filtered wave function (abs)'])

    cmp_func(v_arr, funcs=[wave_func_image_abs, noised_wave_func_filtered_image_abs], caption='Comparison of source and filtered wave function images', title=f'./results/second/{n}/wave_func_image_cmp', legend=['Source wave function image', 'Filtered wave function image'])

    # find frequency response of filter
    z, p, k = filter
    w, h = freqs_zpk(z, p, k, worN=np.linspace(0, F, 1000))
    plot_freq_response(w, abs(h), caption='Frequency response of filter', title=f'./results/second/{n}/filter_frequency_response', scale='linear')
    w_log, h_log = freqs_zpk(z, p, k, worN=np.linspace(0, 10000, 100000))
    plot_freq_response(w_log, abs(h_log), caption='Frequency response of filter (log)', title=f'./results/second/{n}/filter_frequency_response_log', scale='log')


def filter_quotes(file_path, T, n=0):
    if not os.path.exists(f'./results/third/{n}'):
        os.makedirs(f'./results/third/{n}')
    date_array = np.array([])
    price_array = np.array([])
    with open(file_path, 'r') as file:
        header = file.readline()
        for line in file:
            DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL = line.split(';')
            # convert date from ddmmyy to timestamp
            timestamp = datetime.datetime.strptime(DATE, '%d%m%y')
            date_array = np.append(date_array, timestamp)
            price_array = np.append(price_array, float(CLOSE))

    t_arr = np.linspace(0, len(price_array), len(price_array))
    plot_func(t_arr, price_array, title=f'./results/third/{n}/source_quotes', caption='Source quotes of SBER')

    filter = get_first_order_filter(T)
    filtered_price = lsim(filter, price_array, t_arr, X0=price_array[0] * T)[1]
    plot_func(date_array, filtered_price, title=f'./results/third/{n}/filtered_quotes', caption='Filtered quotes of SBER', labels=['Date', 'Price'])
    cmp_func(date_array, funcs=[price_array, filtered_price], caption='Comparison of source and filtered quotes', title=f'./results/third/{n}/quotes_cmp', legend=['Source quotes', 'Filtered quotes'], labels=['Date', 'Price'])



# FIRST TASK 

# first_task(T=10)
# first_task(T=100)



# SECOND TASK 

# filtering(a=4, b=1, c=0, d=0, t1=1, t2=4, L=10, F=10, filter=get_first_order_filter(T=0.5), n=1)
# filtering(a=4, b=1, c=0, d=0, t1=1, t2=4, L=10, F=10, filter=get_first_order_filter(T=0.3), n=2)
# filtering(a=4, b=1, c=0, d=0, t1=1, t2=4, L=10, F=10, filter=get_first_order_filter(T=0.1), n=3)

# filtering(a=10, b=1, c=0, d=0, t1=1, t2=4, L=10, F=10, filter=get_first_order_filter(T=0.3), n=4)
# filtering(a=30, b=1, c=0, d=0, t1=1, t2=4, L=10, F=10, filter=get_first_order_filter(T=0.3), n=5)



# THIRD TASK
# W0 = 80
# A = 30
# filtering(a=4, b=0, c=0.4, d=80, t1=1, t2=4, L=10, F=10, filter=get_second_order_filter(T1=1/W0, T2=A/W0, T3=1/(A * W0)), n=6)

# W0 = 20
# A = 30
# filtering(a=4, b=0, c=0.4, d=80, t1=1, t2=4, L=10, F=10, filter=get_second_order_filter(T1=1/W0, T2=A/W0, T3=1/(A * W0)), n=7)

# W0 = 2000
# A = 300
# filtering(a=4, b=0, c=1, d=80, t1=1, t2=4, L=10, F=10, filter=get_second_order_filter(T1=1/W0, T2=A/W0, T3=1/(A * W0)), n=8)

# W0 = 80
# A = 30
# filtering(a=4, b=0, c=2, d=80, t1=1, t2=4, L=10, F=10, filter=get_second_order_filter(T1=1/W0, T2=A/W0, T3=1/(A * W0)), n=9)



# FOURTH TASK

filter_quotes(file_path='./SBER.csv', T=1, n=1)
filter_quotes(file_path='./SBER.csv', T=7, n=2)
filter_quotes(file_path='./SBER.csv', T=30, n=3)
filter_quotes(file_path='./SBER.csv', T=90, n=4)
filter_quotes(file_path='./SBER.csv', T=356, n=5)



# plt.show()
