import numpy as np
import matplotlib.pyplot as plt
import os 
import time


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
    plt.xlabel('v')
    plt.ylabel('f(v)')
    plt.legend(['Real', 'Imag'], loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def dot_product(X, f, g):
    dx = X[1] - X[0]
    return np.trapz(f * g, dx=dx)


get_fourier_image = lambda X, V, func: np.array([dot_product(X, func, (lambda t: np.e ** (-1j * 2 * np.pi * image_clip * t))(X)) for image_clip in V])
get_fourier_function = lambda X, V, image: np.array([dot_product(V, image, (lambda t: np.e ** (1j * 2 * np.pi * x * t))(V)) for x in X])

get_wave_func = lambda a, t1, t2: np.vectorize(lambda t: a if t1 <= t <= t2 else 0, otypes=[complex])


### Task 1
def dot_prod(steps, T):
    if not os.path.exists(f'plots/{steps}_{T}'):
        os.makedirs(f'plots/{steps}_{T}')

    wave_func = get_wave_func(1, -0.5, 0.5)
    X = np.linspace(-2, 2, steps)
    V = np.linspace(-T, T, steps)

    f = wave_func(X)
    true_image = (lambda v: np.sinc(v))(V)
    num_image = get_fourier_image(X, V, f)

    # images 
    plot_func(X, f, 'Wave function', f'plots/{steps}_{T}/wave_func', legend=['Wave function'], labels=['t', 'f(t)'])
    plot_func(V, true_image, 'True image', f'plots/{steps}_{T}/true_image', legend=['True image'], labels=['v', 'F(v)'])
    plot_image(V, num_image, 'Numerical image', f'plots/{steps}_{T}/num_image')

    cmp_func(V, [true_image, num_image], 'Comparison', f'plots/{steps}_{T}/cmp_images', legend=['True image', 'Numerical image'], labels=['v', 'F(v)'])
    plot_func(V, true_image - num_image, 'Error', f'plots/{steps}_{T}/error', legend=['Difference'], labels=['v', 'F(v)'])

    # restoring 
    num_restored = get_fourier_function(X, V, num_image)
    true_restored = get_fourier_function(X, V, true_image)

    plot_func(X, num_restored, 'Numerical restored function', f'plots/{steps}_{T}/num_restored', legend=['Numerical restored function'], labels=['t', 'f(t)'])
    plot_func(X, true_restored, 'True restored function', f'plots/{steps}_{T}/true_restored', legend=['True restored function'], labels=['t', 'f(t)'])

    cmp_func(X, [f, num_restored], 'Comparison', f'plots/{steps}_{T}/cmp_restored', legend=['Source function', 'Numerical restored function'], labels=['t', 'f(t)'])
    plot_func(X, f - num_restored, 'Error', f'plots/{steps}_{T}/error_restored', legend=['Difference'], labels=['t', 'f(t)'])


def fft(steps):
    if not os.path.exists(f'plots/fft_{steps}'):
        os.makedirs(f'plots/fft_{steps}')

    wave_func = get_wave_func(1, -0.3, 0.5)
    X = np.linspace(-2, 2, steps)
    f = wave_func(X)

    num_image = np.fft.fftshift(np.fft.fft(f)) / np.sqrt(steps)
    V = np.fft.fftshift(np.fft.fftfreq(steps, 1 / steps)) * steps

    # inverse fft
    num_restored = np.fft.ifft(np.fft.ifftshift(num_image)) * np.sqrt(steps)
    plot_func(X, num_restored, 'Numerical restored function', f'plots/fft_{steps}/num_restored', legend=['Numerical restored function'], labels=['t', 'f(t)'])

    cmp_func(X, [f, num_restored], 'Comparison', f'plots/fft_{steps}/cmp_func', legend=['Source function', 'Numerical restored function'], labels=['t', 'f(t)'])
    plot_func(X, f - num_restored, 'Error', f'plots/fft_{steps}/error', legend=['Difference'], labels=['t', 'f(t)'])



# timer = time.time()
# dot_prod(1000, 20)
# print('trapz time:', time.time() - timer)

# timer = time.time()
# dot_prod(1000, 80)
# print('trapz time:', time.time() - timer)

# timer = time.time()
# dot_prod(10000, 80)
# print('trapz time:', time.time() - timer)

# timer = time.time()
fft(1000)
# print('FFT time:', time.time() - timer)

plt.show()