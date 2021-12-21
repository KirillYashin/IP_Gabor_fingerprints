import cv2
import numpy as np
import matplotlib.pyplot as plt


def bgr_2_gray(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


def gabor_filter(k_size=111, sigma=10, gamma=1.2, lmb=10, psi=0, angle=0):
    d = k_size // 2
    gabor = np.zeros((k_size, k_size), dtype=np.float32)

    for y in range(k_size):
        for x in range(k_size):
            px = x - d
            py = y - d
            theta = angle / 180. * np.pi
            _x = np.cos(theta) * px + np.sin(theta) * py
            _y = -np.sin(theta) * px + np.cos(theta) * py
            gabor[y, x] = np.exp(-(_x ** 2 + gamma ** 2 * _y ** 2) / (2 * sigma ** 2)) * np.cos(
                2 * np.pi * _x / lmb + psi)

    gabor /= np.sum(np.abs(gabor))
    return gabor


def gabor_filtering(gray, k_size=111, sigma=10, gamma=1.2, lmb=10, psi=0, angle=0):
    gabor = gabor_filter(k_size=k_size, sigma=sigma, gamma=gamma, lmb=lmb, psi=0, angle=angle)

    out = cv2.filter2D(gray, -1, gabor)
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    return out


def gabor_process(img, params):
    h, w, _ = img.shape
    gray = bgr_2_gray(img).astype(np.float32)
    angles = np.linspace(0, 180, 4)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)
    out = np.zeros([h, w], dtype=np.float32)

    for i, angle in enumerate(angles):
        _out = gabor_filtering(gray, k_size=int(params[0]), sigma=float(params[1]),
                               gamma=float(params[2]), lmb=int(params[3]), angle=angle)

        out += _out

    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out


for i in range(6):
    img = cv2.imread('3.jpg').astype(np.float32)
    out = gabor_process(img, input().split())

    cv2.imwrite(f'out_{i}.jpg', out)
    cv2.imshow('result', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
