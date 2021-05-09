# Input: Input image H, downscaling factor s, patch size np.
# Output: Downscaled image D.
# 1: procedure DOWNSCALEIMAGE
# 2: L ← subSample(convValid(H, P(s)), s)
# 3: L2 ← subSample(convValid(H2, P(s)), s)
# 4: M ← convValid(L, P(√np))
# 5: Sl ← convValid(L2, P(√np)) − M2
# 6: Sh ← convValid(L2, P(√np)) − M2
# 7: R ← √Sh/Sl
# 8: R(Sl < ) ← 0
# 9: N ← convFull(IM, P(√np))
# 10: T ← convFull(R × M, P(√np))
# 11: M ← convFull(M, P(√np))
# 12: R ← convFull(R, P(√np))
# 13: D ← (M + R × L − T)/N

import argparse
import cv2
import numpy as np

def new_size_by_factor(image, factor):
    width = image.shape[1]
    height = image.shape[0]
    (newWidth, newHeight) = round(width/factor), round(height/factor) 
    return (newWidth, newHeight)


def cubic(img, factor):
    (newWidth, newHeight) = new_size_by_factor(img, factor)
    subsampled_img = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_CUBIC)
    return subsampled_img

def subSample(img, factor):
    (newWidth, newHeight) = new_size_by_factor(img, factor)
    subsampled_img = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_AREA)
    return subsampled_img

def convValid(img, ksize):
    # convolvedImg = np.convolve(img, kernel, mode="valid")
    convolvedImg = cv2.blur(img, (2,2))
    return convolvedImg

def convFull(img, ksize):
    # convolvedImg = np.convolve(img, kernel, mode="full")
    convolvedImg = cv2.blur(img, (2,2))
    return convolvedImg

def P(img):
    return img

def downscale_channel(H, s, n):
    e = 0.000001
    L = subSample(convValid(H, P(s)), s)
    L2 = subSample(convValid(cv2.pow(H, 2), P(s)), s)
    M = convValid(L, P(cv2.sqrt(n)))
    print("M")
    print(M)
    Sl = cv2.subtract(convValid(cv2.pow(L, 2), P(cv2.sqrt(n))), cv2.pow(M, 2))
    print("Sl")
    print(Sl)
    Sh = cv2.subtract(convValid(L2, P(cv2.sqrt(n))), cv2.pow(M, 2))
    R = cv2.sqrt(cv2.divide(Sh,Sl))
    print("R")
    print(R)
    # R(Sl < e) = 0
    R = np.where(R < e, 0, R)
    N = L # convFull(IM, P(cv2.sqrt(n)))
    T = convFull(cv2.multiply(R, M), P(cv2.sqrt(n)))
    M = convFull(M, P(cv2.sqrt(n)))
    R = convFull(R, P(cv2.sqrt(n)))
    D = cv2.divide(cv2.subtract(cv2.add(M, cv2.multiply(R, L)), T), 1) # N
    D = np.nan_to_num(0)
    return D

# Input: Input image H, downscaling factor s, patch size np.
# Output: Downscaled image D.
def downscale(H, s, n):
    H = np.float32(H)
    (newWidth, newHeight) = new_size_by_factor(H, factor)
    D = np.ones((newHeight, newWidth, 3))
    print("H="+str(H.shape))
    print("D="+str(D.shape))
    for c in range(0, 3):
        D[:,:,c] = downscale_channel(H[:,:,c], s, n)
    print(D)
    print(D)
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
    
    factor = 2
    subsampled = subSample(image, factor)
    print("subsampled shape = " + str(subsampled.shape))
    cv2.imwrite('./output/subsampled.jpeg', subsampled)

    factor = 2
    cubic = cubic(image, factor)
    print("cubic shape = " + str(cubic.shape))
    cv2.imwrite('./output/cubic.jpeg', cubic)

    downscaled = downscale(image, 2, 4)
    print("downscaled")
    print(downscaled)
    cv2.imwrite('./output/downscaled.jpeg', downscaled)



    

