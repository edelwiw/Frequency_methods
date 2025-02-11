import numpy as np
import matplotlib.pyplot as plt
import os 
import time


def plot_func(X, func, caption, title, legend=['Source function'], labels=['t', 'f(t)'], xlim=10):
    ymin = min(func) 
    ymax = max(func) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.xlim(-xlim, xlim)
    plt.plot(X, func.real)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def cmp_func(X, funcs, caption, title, legend=['Source function', 'Restored function'], labels=['t', 'f(t)'], xlim=10):
    ymin = min([min(func) for func in funcs])
    ymax = max([max(func) for func in funcs])

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.xlim(-xlim, xlim)
    for f in funcs:
     plt.plot(X, f.real, linestyle='dashed' if f == 0 else None)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def cmp_func_diff(X, funcs, caption, title, legend=['Source function', 'Restored function'], labels=['t', 'f(t)'], xlim=10):
    ymin = min([min(func) for func in funcs])
    ymax = max([max(func) for func in funcs])

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.xlim(-xlim, xlim)
    for f in range(len(funcs)):
     plt.plot(X[f], funcs[f].real, linestyle='dashed' if f == 0 else None)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legend, loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def plot_image(X, func, caption, title, xlim=10):
    ymin = min(func.real.min(), func.imag.min())
    ymax = max(func.real.max(), func.imag.max())

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.xlim(-xlim, xlim)
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

get_wave_func = lambda a, t1, t2: np.vectorize(lambda t: a if t1 <= t <= t2 else 0, otypes=[complex])
get_fourier_image = lambda X, V, func: np.array([dot_product(X, func, (lambda t: np.e ** (-1j * 2 * np.pi * image_clip * t))(X)) for image_clip in V])
get_fourier_function = lambda X, V, image: np.array([dot_product(V, image, (lambda t: np.e ** (1j * 2 * np.pi * x * t))(V)) for x in X])

           


### Task 1
def dot_prod(steps, T):
    if not os.path.exists(f'plots/{steps}_{T}'):
        os.makedirs(f'plots/{steps}_{T}')

    wave_func = get_wave_func(1, -0.5, 0.5)
    X = np.linspace(-5, 5, steps)
    V = np.linspace(-T, T, steps)

    f = wave_func(X)
    true_image = (lambda v: np.sinc(v))(V)
    num_image = get_fourier_image(X, V, f)

    # images 
    plot_func(X, f, 'Wave function', f'plots/{steps}_{T}/wave_func', legend=['Wave function'], labels=['t', 'f(t)'], xlim=2)
    plot_func(V, true_image, 'True image', f'plots/{steps}_{T}/true_image', legend=['True image'], labels=['v', 'F(v)'], xlim=10)
    plot_image(V, num_image, 'Numerical image', f'plots/{steps}_{T}/num_image', xlim=10)

    cmp_func(V, [true_image, num_image], 'Comparison', f'plots/{steps}_{T}/cmp_images', legend=['True image', 'Numerical image'], labels=['v', 'F(v)'], xlim=10)
    plot_func(V, true_image - num_image, 'Error', f'plots/{steps}_{T}/error', legend=['Difference'], labels=['v', 'F(v)'], xlim=10)

    # restoring 
    num_restored = get_fourier_function(X, V, num_image)
    true_restored = get_fourier_function(X, V, true_image)

    plot_func(X, num_restored, 'Numerical restored function', f'plots/{steps}_{T}/num_restored', legend=['Numerical restored function'], labels=['t', 'f(t)'], xlim=2)
    # plot_func(X, true_restored, 'True restored function', f'plots/{steps}_{T}/true_restored', legend=['True restored function'], labels=['t', 'f(t)'], xlim=2)

    cmp_func(X, [f, num_restored], 'Comparison', f'plots/{steps}_{T}/cmp_restored', legend=['Source function', 'Numerical restored function'], labels=['t', 'f(t)'], xlim=2)
    plot_func(X, f - num_restored, 'Error', f'plots/{steps}_{T}/error_restored', legend=['Difference'], labels=['t', 'f(t)'], xlim=2)


def dft(steps):
    if not os.path.exists(f'plots/fft_{steps}'):
        os.makedirs(f'plots/fft_{steps}')
    
    # get function 
    wave_func = get_wave_func(1, -0.5, 0.5)
    X = np.linspace(-5, 5, steps)
    f = wave_func(X)

    # calc DFT 
    V_dft = np.fft.fftshift(np.fft.fftfreq(steps, 10 / steps)) 
    image_dft = np.fft.fftshift(np.fft.fft(f, norm='ortho')) 
    plot_image(V_dft, np.sinc(V_dft), 'sinc', '', xlim=20)
    plot_image(V_dft, image_dft, 'DFT', f'plots/fft_{steps}/dft_image', xlim=20)

    # Inverse DFT 
    f_restored = np.fft.ifft(np.fft.ifftshift(image_dft), norm='ortho') 
    plot_func(X, f_restored, 'Numerical restored function', f'plots/fft_{steps}/num_restored', legend=['Numerical restored function'], labels=['t', 'f(t)'], xlim=5)
    cmp_func(X, [f, f_restored], 'Comparison', f'plots/fft_{steps}/cmp_func', legend=['Source function', 'Numerical restored function'], labels=['t', 'f(t)'], xlim=5)
    plot_func(X, f - f_restored, 'Error', f'plots/fft_{steps}/error', legend=['Difference'], labels=['t', 'f(t)'], xlim=5)

    # continues image 
    dt = X[1] - X[0]
    dft_cont_image = image_dft * dt * np.exp(-1j * V_dft * 2 * np.pi * X[0]) * np.sqrt(steps)  
    plot_image(V_dft, dft_cont_image, 'DFT image', f'plots/fft_{steps}/dft_cont_image', xlim=20)
    cmp_func(V_dft, [np.sinc(V_dft), dft_cont_image], 'Comparison', f'plots/fft_{steps}/cmp_cont_image', legend=['True image', 'DFT image'], labels=['v', 'F(v)'], xlim=20)

    # restore from continues image
    dv = V_dft[1] - V_dft[0]
    f_restored_cont = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(dft_cont_image), norm='ortho') * np.sqrt(steps) * dv)
    plot_func(X, f_restored_cont, 'Numerical restored function', f'plots/fft_{steps}/num_restored_cont', legend=['Numerical restored function'], labels=['t', 'f(t)'], xlim=2)

    


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
# dft(10000)
# print('FFT time:', time.time() - timer)

def interpolation(func: callable, dt, steps, B, dir='interpolation', num=0):
    if not os.path.exists(f'plots/{dir}_{num}'):
        os.makedirs(f'plots/{dir}_{num}')

    t_cont = np.linspace(-50, 50, 10000)
    plot_func(t_cont, func(t_cont), 'Source function', f'plots/{dir}_{num}/source_func', legend=['Source function'], labels=['t', 'f(t)'], xlim=7)

    t_sampled = np.arange(-50, 50, dt)
    f_sampled = func(t_sampled)
    plot_func(t_sampled, f_sampled, 'Sampled function', f'plots/{dir}_{num}/sampled_func', legend=['Sampled function'], labels=['t', 'f(t)'], xlim=7)
    cmp_func_diff([t_cont, t_sampled], [func(t_cont), f_sampled], 'Comparison', f'plots/{dir}_{num}/cmp_func', legend=['Source function', 'Sampled function'], labels=['t', 'f(t)'], xlim=7)

    #image 
    V = np.linspace(-20, 20, 1000)
    image = get_fourier_image(t_sampled, V, f_sampled)
    plot_image(V, image, 'Image', f'plots/{dir}_{num}/image', xlim=8)

    interpolation 
    f_interp = np.vectorize(lambda t: np.sum([func(n) * np.sinc(2 * B * (t - n)) for n in t_sampled]))
    t_interp = np.linspace(-10, 10, 1000)
    plot_func(t_interp, f_interp(t_interp), 'Interpolated function', f'plots/{dir}_{num}/interpolated_func', legend=['Interpolated function'], labels=['t', 'f(t)'], xlim=5)
    cmp_func_diff([t_cont, t_interp], [func(t_cont), f_interp(t_interp)], 'Comparison', f'plots/{dir}_{num}/cmp_func_interpolated', legend=['Source function', 'Interpolated function'], labels=['t', 'f(t)'], xlim=5)

    restored_image = get_fourier_image(t_interp, V, f_interp(t_interp))
    plot_image(V, restored_image, 'Restored image', f'plots/{dir}_{num}/restored_image', xlim=8)
    


sinsin = lambda x: 3 * np.sin(2 *np.pi * x + np.pi) + 2 * np.sin(6 * np.pi * x - 5)
sinc = lambda x: np.sinc(2 * x)

interpolation(sinsin, dt=1/8, steps=100, B=4, num=1) 
# interpolation(sinsin, dt=1/4, steps=100, B=6, num=2)
# interpolation(sinsin, dt=1/12, steps=100, B=6, num=3)
# interpolation(sinc, dt=1/4, steps=100, B=2, num=4)
# interpolation(sinc, dt=1/2, steps=100, B=2, num=5)


plt.show()