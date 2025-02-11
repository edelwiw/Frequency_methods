import numpy as np 
import matplotlib.pyplot as plt 
import librosa
import librosa.display


def plot_func(func, t0, t1, caption = '', title = ''):
    # limits
    x_min = t0 #- 0.3 * (t1 - t0)
    x_max = t1 #+ 0.3 * (t1 - t0)

    ymin = min(func(np.linspace(x_min, x_max, 1000))) 
    ymax = max(func(np.linspace(x_min, x_max, 1000))) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    x = np.linspace(x_min, x_max, 1000)
    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(x, func(x).real)
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend(["Source function"], loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')

    
def plot_complex_func(func, t0, t1, caption = '', title = ''):
    # limits
    x_min = t0 #- 0.3 * (t1 - t0)
    x_max = t1 #+ 0.3 * (t1 - t0)

    values = func(np.linspace(x_min, x_max, 1000))
    ymin = min(values.real.min(), values.imag.min())
    ymax = max(values.real.max(), values.imag.max())

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    x = np.linspace(x_min, x_max, 1000)
    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(x, func(x).real)
    plt.plot(x, func(x).imag)
    plt.xlabel('v')
    plt.ylabel('f(v)')
    plt.legend(['Real', 'Imag'], loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def plot_waveform(wave, time, caption = '', title = ''):
    t = np.linspace(0, time - 0.1, 10000)
    plt.figure(figsize=(8, 5)) 

    plt.plot(t, wave(t))
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend(['Source waveform'], loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def plot_wave_image(func, t0, t1, caption = '', title = ''):
    # limits
    x_min = t0 #- 0.3 * (t1 - t0)
    x_max = t1 #+ 0.3 * (t1 - t0)

    values = func(np.linspace(x_min, x_max, 1000))
    ymin = min(values.real.min(), values.imag.min())
    ymax = max(values.real.max(), values.imag.max())

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    x = np.linspace(x_min, x_max, 1000)
    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(x, func(x))
    plt.xlabel('v')
    plt.ylabel('f(v)')
    plt.legend(['Image'], loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid()
    if title != '':
        plt.savefig(title + '.png')


def dot_product(f, g, a, b):
    '''
        Function to calculate dot product of two functions

        f, g - functions
        a, b - limits

        result: dot product of f and g on [a, b]
        
    '''

    x = np.linspace(a, b, 1000)
    dx = x[1] - x[0]
    return np.dot(f(x), g(x)) * dx


def fourier_image(func, a, b):
    image = lambda v: 1 / (np.sqrt(2 * np.pi)) * dot_product(func, lambda t: np.e ** (-1j * v * t), a, b)
    return np.vectorize(image)


def wave_fourier_image(func, a, b):
    image = lambda v: dot_product(func, lambda t: np.e ** (-2 * np.pi * 1j * v * t), a, b)
    return np.vectorize(image)


def parseval_check(func, a, b):
    '''
        Function to check Parseval's identity

        func - source function
        a, b - limits

        result: True if Parseval's theorem is correct, False otherwise
    '''
    f_image = fourier_image(func, a, b)
    f_image_abs = np.vectorize(lambda t: abs(f_image(t)))

    left = abs(dot_product(func, func, a, b))
    right = dot_product(f_image_abs, f_image_abs, -100, 100)
    print(left, right)
    return abs(left - right) < 1e-2



### TASK 1 

# # rectangle function 

def rectangle(t, a, b):
    if abs(t) <= b:
        return a
    return 0

# # rectangle 1 
# rectangle_1 = np.vectorize(lambda t: rectangle(t, a=1, b=0.5), otypes=[np.complex_])
# plot_func(rectangle_1, -5, 5, caption='Rectangle function (a = 1, b = 0.5)', title='media/rectangle_1')

# rectangle_1_image = fourier_image(rectangle_1, -30, 30)
# plot_complex_func(rectangle_1_image, -30, 30, caption='Fourier image of rectangle function (a = 1, b = 0.5)', title='media/rectangle_1_image')

# print(f'Parseval check for rectangle 1: {parseval_check(rectangle_1, a=-0.5, b=0.5)}')

# # rectangle 2
# rectangle_2 = np.vectorize(lambda t: rectangle(t, a=1, b=1), otypes=[np.complex_])
# plot_func(rectangle_2, -5, 5, caption='Rectangle function (a = 1, b = 1)', title='media/rectangle_2')

# rectangle_2_image = fourier_image(rectangle_2, -30, 30)
# plot_complex_func(rectangle_2_image, -30, 30, caption='Fourier image of rectangle function (a = 1, b = 1)', title='media/rectangle_2_image')

# print(f'Parseval check for rectangle 2: {parseval_check(rectangle_2, a=-1, b=1)}')

# # rectangle 3
# rectangle_3 = np.vectorize(lambda t: rectangle(t, a=2, b=2), otypes=[np.complex_])
# plot_func(rectangle_3, -5, 5, caption='Rectangle function (a = 2, b = 2)', title='media/rectangle_3')

# rectangle_3_image = fourier_image(rectangle_3, -30, 30)
# plot_complex_func(rectangle_3_image, -30, 30, caption='Fourier image of rectangle function (a = 2, b = 2)', title='media/rectangle_3_image')

# print(f'Parseval check for rectangle 3: {parseval_check(rectangle_3, a=-2, b=2)}')



# triangle function

# def triangle(t, a, b):
#     if abs(t) <= b:
#         return a - abs(a * t / b)
#     return 0


# triangle 1
# triangle_1 = np.vectorize(lambda t: triangle(t, a=1, b=0.5), otypes=[np.complex_])
# plot_func(triangle_1, -5, 5, caption='Triangle function (a = 1, b = 0.5)', title='media/triangle_1')

# triangle_1_image = fourier_image(triangle_1, -30, 30)
# plot_complex_func(triangle_1_image, -30, 30, caption='Fourier image of triangle function (a = 1, b = 0.5)', title='media/triangle_1_image')

# print(f'Parseval check for triangle 1: {parseval_check(triangle_1, a=-0.5, b=0.5)}')

# triangle 2
# triangle_2 = np.vectorize(lambda t: triangle(t, a=1, b=1), otypes=[np.complex_])
# plot_func(triangle_2, -5, 5, caption='Triangle function (a = 1, b = 1)', title='media/triangle_2')

# triangle_2_image = fourier_image(triangle_2, -30, 30)
# plot_complex_func(triangle_2_image, -30, 30, caption='Fourier image of triangle function (a = 1, b = 1)', title='media/triangle_2_image')

# print(f'Parseval check for triangle 2: {parseval_check(triangle_2, a=-1, b=1)}')

# triangle 3
# triangle_3 = np.vectorize(lambda t: triangle(t, a=2, b=2), otypes=[np.complex_])
# plot_func(triangle_3, -5, 5, caption='Triangle function (a = 2, b = 2)', title='media/triangle_3')

# triangle_3_image = fourier_image(triangle_3, -30, 30)
# plot_complex_func(triangle_3_image, -30, 30, caption='Fourier image of triangle function (a = 2, b = 2)', title='media/triangle_3_image')

# print(f'Parseval check for triangle 3: {parseval_check(triangle_3, a=-2, b=2)}')



# sinc function

def sinc(t, a, b):
    return a * np.sinc(b * t)

# sinc 1
sinc_1 = np.vectorize(lambda t: sinc(t, a=1, b=0.5), otypes=[np.complex_])
plot_func(sinc_1, -30, 30, caption='Sinc function (a = 1, b = 0.5)', title='media/sinc_1')

sinc_1_image = fourier_image(sinc_1, -5, 5)
plot_complex_func(sinc_1_image, -5, 5, caption='Fourier image of sinc function (a = 1, b = 0.5)', title='media/sinc_1_image')

# print(f'Parseval check for sinc 1: {parseval_check(sinc_1, a=-100, b=100)}')

# sinc 2
sinc_2 = np.vectorize(lambda t: sinc(t, a=1, b=1), otypes=[np.complex_])
plot_func(sinc_2, -30, 30, caption='Sinc function (a = 1, b = 1)', title='media/sinc_2')

sinc_2_image = fourier_image(sinc_2, -5, 5)
plot_complex_func(sinc_2_image, -5, 5, caption='Fourier image of sinc function (a = 1, b = 1)', title='media/sinc_2_image')

# print(f'Parseval check for sinc 2: {parseval_check(sinc_2, a=-100, b=100)}')

# sinc 3
sinc_3 = np.vectorize(lambda t: sinc(t, a=2, b=0.5), otypes=[np.complex_])
plot_func(sinc_3, -30, 30, caption='Sinc function (a = 2, b = 2)', title='media/sinc_3')

sinc_3_image = fourier_image(sinc_3, -5, 5)
plot_complex_func(sinc_3_image, -10, 10, caption='Fourier image of sinc function (a = 2, b = 2)', title='media/sinc_3_image')

# print(f'Parseval check for sinc 3: {parseval_check(sinc_3, a=-100, b=100)}')

# # gaussian function

# def gaussian(t, a, b):
#     return a * np.e ** (-b * t ** 2)

# # gaussian 1
# gaussian_1 = np.vectorize(lambda t: gaussian(t, a=1, b=0.5), otypes=[np.complex_])
# plot_func(gaussian_1, -5, 5, caption='Gaussian function (a = 1, b = 0.5)', title='media/gaussian_1')

# gaussian_1_image = fourier_image(gaussian_1, -30, 30)
# plot_complex_func(gaussian_1_image, -30, 30, caption='Fourier image of gaussian function (a = 1, b = 0.5)', title='media/gaussian_1_image')

# print(f'Parseval check for gaussian 1: {parseval_check(gaussian_1, a=-100, b=100)}')

# # gaussian 2
# gaussian_2 = np.vectorize(lambda t: gaussian(t, a=1, b=1), otypes=[np.complex_])
# plot_func(gaussian_2, -5, 5, caption='Gaussian function (a = 1, b = 1)', title='media/gaussian_2')

# gaussian_2_image = fourier_image(gaussian_2, -30, 30)
# plot_complex_func(gaussian_2_image, -30, 30, caption='Fourier image of gaussian function (a = 1, b = 1)', title='media/gaussian_2_image')

# print(f'Parseval check for gaussian 2: {parseval_check(gaussian_2, a=-100, b=100)}')

# # gaussian 3
# gaussian_3 = np.vectorize(lambda t: gaussian(t, a=2, b=2), otypes=[np.complex_])
# plot_func(gaussian_3, -5, 5, caption='Gaussian function (a = 2, b = 2)', title='media/gaussian_3')

# gaussian_3_image = fourier_image(gaussian_3, -30, 30)
# plot_complex_func(gaussian_3_image, -30, 30, caption='Fourier image of gaussian function (a = 2, b = 2)', title='media/gaussian_3_image')

# print(f'Parseval check for gaussian 3: {parseval_check(gaussian_3, a=-100, b=100)}')



# # fade function

# def fade(t, a, b):
#     return a * np.e ** (-b * abs(t))

# # fade 1
# fade_1 = np.vectorize(lambda t: fade(t, a=1, b=0.5), otypes=[np.complex_])
# plot_func(fade_1, -5, 5, caption='Fade function (a = 1, b = 0.5)', title='media/fade_1')

# fade_1_image = fourier_image(fade_1, -30, 30)
# plot_complex_func(fade_1_image, -30, 30, caption='Fourier image of fade function (a = 1, b = 0.5)', title='media/fade_1_image')

# print(f'Parseval check for fade 1: {parseval_check(fade_1, a=-100, b=100)}')

# # fade 2
# fade_2 = np.vectorize(lambda t: fade(t, a=1, b=1), otypes=[np.complex_])
# plot_func(fade_2, -5, 5, caption='Fade function (a = 1, b = 1)', title='media/fade_2')

# fade_2_image = fourier_image(fade_2, -30, 30)
# plot_complex_func(fade_2_image, -30, 30, caption='Fourier image of fade function (a = 1, b = 1)', title='media/fade_2_image')

# print(f'Parseval check for fade 2: {parseval_check(fade_2, a=-100, b=100)}')

# # fade 3
# fade_3 = np.vectorize(lambda t: fade(t, a=2, b=2), otypes=[np.complex_])
# plot_func(fade_3, -5, 5, caption='Fade function (a = 2, b = 2)', title='media/fade_3')

# fade_3_image = fourier_image(fade_3, -30, 30)
# plot_complex_func(fade_3_image, -30, 30, caption='Fourier image of fade function (a = 2, b = 2)', title='media/fade_3_image')

# print(f'Parseval check for fade 3: {parseval_check(fade_3, a=-100, b=100)}')



### TASK 2 

# moved rectangle 1 
# moved_rectangle_1 = np.vectorize(lambda t: rectangle(t - 1, a=1, b=0.5), otypes=[np.complex_])
# plot_func(moved_rectangle_1, -5, 5, caption='Moved rectangle function (a = 1, b = 0.5, c = -1)', title='media/moved_rectangle_1')

# moved_rectangle_1_image = fourier_image(moved_rectangle_1, -30, 30)
# plot_complex_func(moved_rectangle_1_image, -30, 30, caption='Fourier image of moved rectangle function (a = 1, b = 0.5, c = -1)', title='media/moved_rectangle_1_image')
# moved_rectangle_1_image_abs = lambda t: abs(moved_rectangle_1_image(t))
# plot_func(moved_rectangle_1_image_abs, -30, 30, caption='Fourier image of moved rectangle function (a = 1, b = 0.5, c = -1)', title='media/moved_rectangle_1_image_abs')

# print(f'Parseval check for moved rectangle 1: {parseval_check(moved_rectangle_1, a=0.5, b=1.5)}')

# # moved rectangle 2
# moved_rectangle_2 = np.vectorize(lambda t: rectangle(t + 1, a=1, b=0.5), otypes=[np.complex_])
# plot_func(moved_rectangle_2, -5, 5, caption='Moved rectangle function (a = 1, b = 0.5, c = 1)', title='media/moved_rectangle_2')

# moved_rectangle_2_image = fourier_image(moved_rectangle_2, -30, 30)
# plot_complex_func(moved_rectangle_2_image, -30, 30, caption='Fourier image of moved rectangle function (a = 1, b = 0.5, c = 1)', title='media/moved_rectangle_2_image')
# moved_rectangle_2_image_abs = lambda t: abs(moved_rectangle_2_image(t))
# plot_func(moved_rectangle_2_image_abs, -30, 30, caption='Fourier image of moved rectangle function (a = 1, b = 0.5, c = 1)', title='media/moved_rectangle_2_image_abs')

# print(f'Parseval check for moved rectangle 2: {parseval_check(moved_rectangle_2, a=-1.5, b=-0.5)}')

# # moved rectangle 3
# moved_rectangle_3 = np.vectorize(lambda t: rectangle(t + 2, a=1, b=0.5), otypes=[np.complex_])
# plot_func(moved_rectangle_3, -5, 5, caption='Moved rectangle function (a = 1, b = 1, c = 2)', title='media/moved_rectangle_3')

# moved_rectangle_3_image = fourier_image(moved_rectangle_3, -30, 30)
# plot_complex_func(moved_rectangle_3_image, -30, 30, caption='Fourier image of moved rectangle function (a = 1, b = 1, c = 2)', title='media/moved_rectangle_3_image')
# moved_rectangle_3_image_abs = lambda t: abs(moved_rectangle_3_image(t))
# plot_func(moved_rectangle_3_image_abs, -30, 30, caption='Fourier image of moved rectangle function (a = 1, b = 1, c = 2)', title='media/moved_rectangle_3_image_abs')

# print(f'Parseval check for moved rectangle 3: {parseval_check(moved_rectangle_3, a=-3, b=-2)}')


### TASK 3 

# read mp3 file as waveform
# wave_from_sample, sr = librosa.load('Chord23.mp3')

# wave_from_time = np.vectorize(lambda t: wave_from_sample[int(t * sr)])
# time = len(wave_from_sample) / sr - 0.001

# plot_waveform(wave_from_time, time, caption='Waveform of Chord23', title='media/waveform')

# # fourier transform
# wave_image = wave_fourier_image(wave_from_time, 0, 0.1)
# wave_image_abs = lambda t: abs(wave_image(t))
# plot_wave_image(wave_image_abs, 0, 4000, caption='Fourier image of Chord23', title='media/wave_image')


plt.show()
