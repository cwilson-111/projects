import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import imageio

def laplacian(shape):
    rows,cols = shape
    r,c = np.mgrid[:rows, :cols]
    center_r, center_c = rows//2, cols // 2

    kernel_freq = (2* np.pi * np.sqrt((r-center_r)**2 + (c-center_c)**2))**2 #freq domain rep of laplacian approx

    kernel_freq = kernel_freq / np.max(kernel_freq) #nomrlaizing kernel
    return kernel_freq

def apply(image_path):
    img = imageio.imread(image_path, mode='F') #load image

    f = fftpack.fft2(img) #2d fft
    fshift = fftpack.fftshift(f) #2d fft

    kernel_freq = laplacian(img.shape) #creating laplacian kernel in freq domain
    filtered_fshift = fshift * kernel_freq #multiply fourier coef w/ lapalcian kernel

    f_ishift = fftpack.ifftshift(filtered_fshift)
    img_rec= np.real(fftpack.ifft2(f_ishift))

    ##showing images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(img_rec, cmap='gray')
    plt.title('Laplacian')

    plt.tight_layout()
    plt.show()

img = img = r"C:\Users\danet\OneDrive\Desktop\Image Processing\Homework\2000949312\astronaut.png"
apply(img)