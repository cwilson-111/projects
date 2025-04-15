import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import imageio.v2 as imageio 

def reconstruct(img_path, remove_high_freq = False):
     
     img = imageio.imread(img_path, mode='F')
     img_array = np.array(img)
    
     f= fftpack.fft2(img_array)
     fshift = fftpack.fftshift(f)

     amp = np.abs(fshift)
     phase = np.angle(fshift)
     if remove_high_freq:
          rows,cols = img_array.shape
          crow,ccol = rows // 2, cols // 2
          cutoff = 50
          fshift[crow - cutoff:crow + cutoff, ccol - cutoff:ccol - cutoff] = 0
     f_ishift = fftpack.ifftshift(fshift)
     img_reconstructed = np.real(fftpack.ifft2(f_ishift))

     error = np.mean((img_array - img_reconstructed) ** 2)

     plt.figure(figsize=(12, 6))

     plt.subplot(1, 3, 1)
     plt.imshow(img_array, cmap='gray')
     plt.title('Original Image')

     plt.subplot(1, 3, 2)
     plt.imshow(img_reconstructed, cmap='gray')
     plt.title('Reconstructed Image')

     plt.subplot(1, 3, 3)
     plt.imshow(np.abs(img_array - img_reconstructed), cmap='gray')
     plt.title('Absolute Error')

     plt.tight_layout()
     plt.show()

img =  r"C:\Users\danet\OneDrive\Desktop\Image Processing\Homework\2000949312\astronaut.png"
reconstruct(img)
reconstruct(img, remove_high_freq=True) #reconstructuing with high freq


