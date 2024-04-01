import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import time
import os 


def plot_func(X, func, caption, title, legend=['Source function']):
    ymin = min(func) 
    ymax = max(func) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(X, func.real)
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend(legend, loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def cmp_func(X, func1, func2, caption, title, legend=['Source function', 'Restored function']):
    ymin = min(min(func1), min(func2))
    ymax = max(max(func1), max(func2)) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(X, func1.real)
    plt.plot(X, func2.real)
    plt.xlabel('t')
    plt.ylabel('f(t)')
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

clip_outer_image_values = lambda image_clip: lambda X, func: np.array([func[i] if abs(X[i]) <= image_clip else 0 for i in range(len(func))])
clip_inner_image_values = lambda image_clip: lambda X, func: np.array([func[i] if abs(X[i]) >= image_clip else 0 for i in range(len(func))])
clip_delta_image_values = lambda pivot, delta: lambda X, func: np.array([0 if (pivot - delta <= X[i] <= pivot + delta  or pivot - delta <= -X[i] <= pivot + delta)else func[i] for i in range(len(func))])



def calc(a, b, c, d, t1, t2, T, image_limits, clip_function=clip_outer_image_values(image_clip=5), n=0):
    if not os.path.exists(f'./results/{n}'):
        os.makedirs(f'./results/{n}')
    X = np.linspace(-T/2, T/2, 1000) # get x values array
    wave = get_wave_func(a, t1, t2)(X) # get wave function values

    noised_wave = apply_noise_to_values(X, wave, b=b, c=c, d=d) # get noised wave function values

    plot_func(X, wave, f'Wave function', f'./results/{n}/wave_function', legend=["Source function"]) # source function 
    plot_func(X, noised_wave, f'Wave function with noise', f'./results/{n}/noised_wave_function', legend=["Noised function"]) # noised function

    V = np.linspace(-image_limits, image_limits, 1000)

    wave_image = get_fourier_image(X, V, wave)
    noised_wave_image = get_fourier_image(X, V, noised_wave)

    plot_image(V, wave_image, f'Wave function image', f'./results/{n}/wave_function_image') # source function image
    plot_image(V, noised_wave_image, f'Wave function image with noise', f'./results/{n}/noised_wave_function_image') # noised function image

    noised_wave_image_clipped = clip_function(V, noised_wave_image) 
    plot_image(V, noised_wave_image_clipped, f'Wave function image with noise clipped', f'./results/{n}/noised_wave_function_image_clipped') # noised function image clipped

    noised_wave_restored = get_fourier_function(X, V, noised_wave_image)
    plot_func(X, noised_wave_restored, f'Wave function restored', f'./results/{n}/wave_function_restored', legend=["Restored function"]) # restored function

    noised_wave_clipped_restored = get_fourier_function(X, V, noised_wave_image_clipped)
    plot_func(X, noised_wave_clipped_restored, f'Wave function de-noised', f'./results/{n}/wave_function_clipped_restored', legend=["De-noised function"]) # restored clipped function

    noised_wave_clipped_restored_image = get_fourier_image(X, V, noised_wave_clipped_restored)
    plot_image(V, noised_wave_clipped_restored_image, f'Wave function clipped image', f'./results/{n}/wave_function_clipped_restored_image') # restored clipped function image

    # plot abs img 
    plot_image(V, np.abs(wave_image), f'Wave function image abs', f'./results/{n}/wave_function_image_abs') # source function image abs
    plot_image(V, np.abs(noised_wave_clipped_restored_image), f'Wave function clipped image abs', f'./results/{n}/wave_function_clipped_restored_image_abs') # restored clipped function image abs

    cmp_func(X, wave, noised_wave_clipped_restored, f'Wave function comparison', f'./results/{n}/wave_function_comparison', legend=["Source function", "De-noised function"]) # source and restored clipped function comparison

# 
# calc(a=3, b=1, c=0, d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=10), n=1)
# calc(a=3, b=1, c=0, d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=5), n=2)
# calc(a=3, b=1, c=0, d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=2), n=3)
# calc(a=3, b=1, c=0, d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=20), n=4)
# calc(a=3, b=0.5, c=0, d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=10), n=5)
# calc(a=3, b=2, c=0, d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=10), n=6)
# calc(a=3, b=0.2, c=0, d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=10), n=7)
# calc(a=3, b=4, c=0, d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=10), n=8)

# calc(a=3, b=0.5, c=0.8, d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=15), n=9)
# clip_image_combine = lambda X, func: clip_outer_image_values(image_clip=15)(X, clip_delta_image_values(pivot=8, delta=0.7)(X, func))
# calc(a=3, b=0.5, c=0.8, d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_image_combine, n=10)
# calc(a=3, b=0, c=1, d=10, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_delta_image_values(pivot=10, delta=1), n=11)

# clip_image_combine = lambda X, func: clip_outer_image_values(image_clip=15)(X, clip_delta_image_values(pivot=10, delta=1)(X, func))
# calc(a=3, b=2, c=1, d=10, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_image_combine, n=12)

# clip_image_combine = lambda X, func: clip_outer_image_values(image_clip=15)(X, clip_delta_image_values(pivot=10, delta=1)(X, func))
# calc(a=3, b=2, c=1, d=10, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_image_combine, n=13)

# clip_image_combine = lambda X, func: clip_outer_image_values(image_clip=15)(X, clip_delta_image_values(pivot=5, delta=1)(X, func))
# calc(a=3, b=0.5, c=1, d=5, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_image_combine, n=14)

calc(a=3, b=0.3, c=1, d=10, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_inner_image_values(image_clip=8), n=15)


# plt.show()


