import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import imageio.v2 as imageio 

def create_gaussian(shape, center, sigma):
    rows,cols= shape
    r,c = np.mgrid[:rows, :cols]
    gaussian = np.exp(-((r - center[0])**2 + (c - center[1])**2) / (2.0 * sigma**2))
    return gaussian

def apply_gaussian(img_path):
    img_array = imageio.imread(img_path, mode = 'F')
    f = fftpack.fft2(img_array)
    fshift = fftpack.fftshift(f)
    
    rows,cols = img_array.shape
    center = (rows // 2, cols // 2)
    sigma = 1000 
    gaussian_filter = create_gaussian(fshift.shape, center,sigma)

    filtered_fshift = fshift * gaussian_filter

    fshift = fftpack.ifftshift(filtered_fshift)
    img_reconstruction = np.real(fftpack.ifft2(filtered_fshift))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(img_reconstruction, cmap='gray')
    plt.title('Filtered Image (Gaussian)')

    plt.tight_layout()
    plt.show()


img = r"C:\Users\danet\OneDrive\Desktop\Image Processing\Homework\2000949312\astronaut.png"
apply_gaussian(img)
