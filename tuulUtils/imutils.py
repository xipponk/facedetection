from matplotlib import pyplot as plt
import numpy as np
import cv2

def translate(image, x, y):
    M = np.float32([[1,0,x], [0,1,y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[0:2]
    if center is None:
        center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w,h))
    return rotated

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[0:2]
    
    if width is None and height is None:
        return image
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def plot_histogram(image, title, mask = None):
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set(title=title, xlabel='Bins', ylabel='# of Pixels', xlim=[0, 256]);
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        ax.plot(hist, color=color)

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged