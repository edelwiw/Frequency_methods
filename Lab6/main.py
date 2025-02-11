import numpy as np
import matplotlib.pyplot as plt
import os 
import time
import cv2 as cv2


def read_image(path):
    # read image and convert it to numpy array
    img = plt.imread(path)
    return img


def show_image(img, title='output', cmap=None):
    plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=70)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imsave(f'{title}.png', img, cmap='gray')


def fourier_transform(img):
    # compute the 2D fourier transform of the image
    img_fourier_r = np.fft.fftshift(np.fft.fft2(img[:, :, 0]))
    img_fourier_g = np.fft.fftshift(np.fft.fft2(img[:, :, 1]))
    img_fourier_b = np.fft.fftshift(np.fft.fft2(img[:, :, 2]))

    img_fourier = np.stack([img_fourier_r, img_fourier_g, img_fourier_b], axis=2)

    # split to abs and phase
    abs_fourier = np.abs(img_fourier)
    phase_fourier = np.angle(img_fourier)

    abs_fourier_log = np.log(abs_fourier + 1)
    # normalize the abs_fourier_log
    
    min_val = np.min(abs_fourier_log)
    scale_factor = (np.max(abs_fourier_log) - np.min(abs_fourier_log))
    abs_fourier_log = (abs_fourier_log - min_val) / scale_factor

    return abs_fourier_log, phase_fourier, min_val, scale_factor


def restore_image(abs_fourier_log, phase_fourier, min_val, scale_factor):
    # restore the image from the fourier transform
    abs_fourier = np.exp(abs_fourier_log * scale_factor + min_val)
    img_fourier = abs_fourier * np.exp(1j * phase_fourier)

    img_restored_r = abs(np.fft.ifft2(np.fft.ifftshift(img_fourier[:, :, 0]))) 
    img_restored_g = abs(np.fft.ifft2(np.fft.ifftshift(img_fourier[:, :, 1])))
    img_restored_b = abs(np.fft.ifft2(np.fft.ifftshift(img_fourier[:, :, 2])))

    img_restored = np.stack([img_restored_r, img_restored_g, img_restored_b], axis=2)


    img_restored = np.clip(img_restored, 0, 1)
    return img_restored


# img = read_image('8.png')

# show_image(img)

# abs_fourier_log, phase_fourier, min_val, scale_factor = fourier_transform(img)
# show_image(abs_fourier_log, 'abs_fourier_log')
# # save to a file
# if os.path.exists('fixed.png'):
#     abs_fourier_log = read_image('fixed.png')
#     print('fixed image loaded')

# img_restored = restore_image(abs_fourier_log, phase_fourier, min_val, scale_factor)
# show_image(img_restored, 'restored')



### TASK 2 

img = read_image('img.jpg')
img_bw = np.mean(img, axis=2)

show_image(img_bw, 'bw')

# narr = [3, 5, 7, 11, 21]
# for n in narr:
#     # create a mask
#     mask_mean = np.ones([n, n]) / (n * n)

#     # apply the mask
#     img_transformed = cv2.filter2D(img_bw, -1, mask_mean)

#     show_image(img_transformed, f'block_{n}')

#     mask_gaussian = np.zeros_like(mask_mean)
#     for i in range(n):
#         for j in range(n):
#             mask_gaussian[i, j] = np.e ** ((-9 / n ** 2) * ((i - (n + 1) / 2) ** 2) + ((j - (n + 1) / 2) ** 2))
#     mask_gaussian = mask_gaussian / np.sum(mask_gaussian)

#     mask_gaussian = mask_gaussian @ mask_gaussian.T

#     img_transformed = cv2.filter2D(img_bw, -1, mask_gaussian)
#     show_image(img_transformed, f'gaussian_{n}')

# def conv(img, mask):
#     for i in range(1, img.shape[0] - 1):
#         for j in range(1, img.shape[1] - 1):
#             img[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * mask)
#     return img

# mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# img_transformed = cv2.filter2D(img_bw, -2, mask)
# show_image(img_transformed, 'laplacian')

# edge detection
mask_x = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

img_transformed = cv2.filter2D(img_bw, -1, mask_x)
show_image(img_transformed, 'edge_x')

plt.show()