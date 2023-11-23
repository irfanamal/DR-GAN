import os
import numpy as np
from glob import glob
import cv2
from matplotlib import pyplot as plt
import math
import random

img_w = 256
img_h = 256
img_channels = 3

def plot_images(images, save2file, path, name, step):
    filename = path + "%d_" % step + name + ".png"

    plt.figure(figsize = (10, 10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        image = images[i, :, :]
        image = np2img(image)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()
    
def save_all_weights(g, d, save_dir, epoch_number, loss):
    for file in os.listdir(save_dir):
        if file.endswith('.h5'):
            os.remove(os.path.join(save_dir, file))
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}_{}.h5'.format(epoch_number, loss)), True)

def preprocess(image):
    max_size = max(image.shape[:2])
    image = cv2.copyMakeBorder(image, math.ceil((max_size - image.shape[0]) / 2),  math.floor((max_size - image.shape[0]) / 2), math.ceil((max_size - image.shape[1]) / 2),  math.floor((max_size - image.shape[1]) / 2), cv2.BORDER_CONSTANT, 0)
    image = cv2.resize(image, (256, 256))
    return image

def get_data(batch_size, test_num, path):
    path1 = path + "drgan/*.*" # distorted image
    path2 = path + "original/*.*"  # gt rectified image      
    loc_list1 = sorted(glob(path1))
    loc_list2 = sorted(glob(path2))

    indexes = random.sample(range(len(loc_list1)), test_num)
    loc_list1 = np.array(loc_list1)[indexes]
    loc_list2 = np.array(loc_list2)[indexes]
    
    img_num = len(loc_list1)
    n_batches = math.ceil(img_num / batch_size)
    
    for i in range(n_batches):
        step_size = batch_size
        if i == n_batches - 1:
            step_size = img_num - i * batch_size

        src = np.zeros((step_size, img_w, img_h, img_channels)) 
        gt = np.zeros((step_size, img_w, img_h, img_channels)) 
        for j in range(step_size):
            # image reading
            img1 = cv2.imread(loc_list1[i * batch_size + j])
            img2 = cv2.imread(loc_list2[i * batch_size + j])
                    
            img1 = np.reshape(img1, (img_w, img_h, img_channels))
            img2 = preprocess(img2)
            
            src[j, :, :] = img1
            gt[j, :, :] = img2
    
        src = src.astype('float32')
        src = (src - 127.5) / 127.5
        gt = gt.astype('float32')
        gt = (gt - 127.5) / 127.5
        yield src, gt

def euclidean_dist(x0, y0, x1, y1):
    return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)

def f(x, y, xc, yc, a1, a2, a3, a4):
    xu = x + (x-xc) * (a1 * euclidean_dist(xc, yc, x, y)**2 + a2 * euclidean_dist(xc, yc, x, y)**4 + a3 * euclidean_dist(xc, yc, x, y)**6 + a4 * euclidean_dist(xc, yc, x, y)**8)
    yu = y + (y-yc) * (a1 * euclidean_dist(xc, yc, x, y)**2 + a2 * euclidean_dist(xc, yc, x, y)**4 + a3 * euclidean_dist(xc, yc, x, y)**6 + a4 * euclidean_dist(xc, yc, x, y)**8)
    return xu, yu

def distort(image):
    distorted = np.zeros_like(image)
    xc, yc = random.random()*255, random.random()*255
    a1 = random.uniform(-1e-5, -1e-8)
    a2 = random.uniform(1e-12, 1e-9)
    a3 = random.uniform(1e-16, 1e-13)
    a4 = random.uniform(1e-20, 1e-17)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x, y = f(j, i, xc, yc, a1, a2, a3, a4)
            if x >= 0 and y >= 0:
                try:
                    distorted[i][j] = image[int(y)][int(x)]
                except:
                    continue
    return distorted

def load_batch(batch_size, train_num, path):   
    path2 = path + '*.*'  # gt rectified image
    loc_list2 = glob(path2)
    
    indexes = random.sample(range(len(loc_list2)), train_num)
    loc_list2 = np.array(loc_list2)[indexes]
    
    train_num = len(loc_list2)
    n_batches = math.ceil(train_num / batch_size)
    
    for i in range(n_batches):
        step_size = batch_size
        if i == n_batches - 1:
            step_size = train_num - i * batch_size

        src = np.zeros((step_size, img_w, img_h, img_channels)) 
        gt = np.zeros((step_size, img_w, img_h, img_channels))   
        for j in range(step_size):
            # image reading
            img2 = cv2.imread(loc_list2[i * batch_size + j])
            img2 = preprocess(img2)
            img1 = distort(img2)
            
            src[j, :, :] = img1
            gt[j, :, :] = img2
        
        src = src.astype('float32')
        src = (src - 127.5) / 127.5
        gt = gt.astype('float32')
        gt = (gt - 127.5) / 127.5
        yield src, gt


def np2img(img):
    img = img * 127.5 + 127.5
    img = img.astype('uint8')
    img = np.reshape(img, [img_w, img_h, img_channels])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img