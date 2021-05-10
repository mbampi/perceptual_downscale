import argparse
import cv2
import numpy as np
import math

def new_size_by_factor(image, factor):
    width = image.shape[1]
    height = image.shape[0]
    (newWidth, newHeight) = round(width/factor), round(height/factor) 
    return (newWidth, newHeight)

def bicubic(img, factor):
    (newWidth, newHeight) = new_size_by_factor(img, factor)
    subsampled_img = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_CUBIC)
    return subsampled_img

def subSample(img, factor):
    (newWidth, newHeight) = new_size_by_factor(img, factor)
    subsampled_img = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_NEAREST)
    return subsampled_img

def convValid(img, ksize):
    ksize = (int(ksize), int(ksize))
    convolvedImg = cv2.blur(img, ksize)
    return convolvedImg

def convFull(img, ksize):
    ksize = (int(ksize), int(ksize))
    convolvedImg = cv2.blur(img, ksize)
    return convolvedImg

def downscale_channel(img, factor, patch_size):
    e = 0.000001
    
    L = subSample(convValid(img, factor), factor)
    L2 = subSample(convValid(cv2.pow(img, 2), factor), factor)
    M = convValid(L, math.sqrt(patch_size))

    Sl = cv2.subtract(convValid(cv2.pow(L, 2), math.sqrt(patch_size)), cv2.pow(M, 2))
    Sh = cv2.subtract(convValid(L2, math.sqrt(patch_size)), cv2.pow(M, 2))

    R = cv2.sqrt(cv2.divide(Sh, Sl))
    R = np.where(R < e, 0, R)
    
    w, h = new_size_by_factor(img, factor)
    IM = np.ones((h, w), dtype=np.float64)

    N = convFull(IM, math.sqrt(patch_size))

    T = convFull(cv2.multiply(R, M), math.sqrt(patch_size))

    M = convFull(M, math.sqrt(patch_size))

    R = convFull(R, math.sqrt(patch_size))

    D = cv2.divide(cv2.subtract(cv2.add(M, cv2.multiply(R, L)), T), N)
    D = np.nan_to_num(D, 0)
    # print("D="+str(D.shape)+"="+str(D))

    return D


# Input: Input image H, downscaling factor s, patch size np.
# Output: Downscaled image D.
def downscale(img, factor, patch_size):
    img = np.float64(img)
    (newWidth, newHeight) = new_size_by_factor(img, factor)
    D = np.ones((newHeight, newWidth, 3))

    for c in range(0, 3):
        D[:,:,c] = downscale_channel(img[:,:,c], factor, patch_size)

    return D


if __name__ == '__main__':
    print("Perceptually Based Downscale")

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
    imgPath = args["image"]
    print("img path = " + imgPath)

    image = cv2.imread(imgPath)
    print("image shape = " + str(image.shape) + " type="+str(type(image)))
    cv2.imwrite('./output/original.jpeg', image)

    factor = 4
    patch_size = 1024
    
    subsampled_img = subSample(image, factor)
    print("subsampled_img shape = " + str(subsampled_img.shape))
    cv2.imwrite('./output/subsampled.jpeg', subsampled_img)

    bicubic_img = bicubic(image, factor)
    print("bicubic_img shape = " + str(bicubic_img.shape))
    cv2.imwrite('./output/bicubic.jpeg', bicubic_img)

    downscaled_img = downscale(image, factor, patch_size)
    print("downscaled_img shape = " + str(downscaled_img.shape))
    cv2.imwrite('./output/perceptual.jpeg', downscaled_img)
