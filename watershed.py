# -*- coding: utf-8 -*-
import cv2
from skimage import morphology, feature, filters
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np


def water(image, image_rgb, image_labels, index, type_trainortest):
    # 滤波     过滤噪声
    #中值滤波器（median): 返回图像局部区域内的中值，用此中值取代该区域内所有像素值。
    denoised = filters.rank.median(image, morphology.disk(5))
    # 梯度计算
    #返回图像的局部梯度值（最大值 - 最小值），用此梯度值代替区域内所有像素值。
    markers_t = filters.rank.gradient(denoised, morphology.disk(5))##半径为5的圆形滤波器
    ax_img = plt.subplot(2, 2, 1)
    ax_img.set_title("gradient")
    ax_img.imshow(markers_t, 'gray')
    print("markers_t")
    print(markers_t)
    # 显示直方图
    ax_hist = plt.subplot(2, 2, 2)
    ax_hist.set_title('hist')
    ax_hist.hist(markers_t.ravel(), bins=256) #ravel将多维数组降为一维
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    # 梯度计算并选取梯度小的区域作为初始区域    将梯度(中值)值低于20的作为开始标记点
    markers = filters.rank.gradient(denoised, morphology.disk(3)) < 10
    print("markers111")
    print(markers)

    # 根据连通性对局部区域标记
    markers = ndi.label(markers)[0]
    print("matkers222")
    print(markers)

    # 梯度计算
    gradient = filters.rank.gradient(denoised, morphology.disk(3))

    # 根据标记的局部区域进行分水岭分割
    labels = morphology.watershed(gradient, markers, mask=image)
    print("labels:")
    #print(labels)
    print(labels.shape[0])
    print(labels.shape[1])
    # print(labels.shape, labels.max())

    # 区域分割效果图转换为边缘提取效果图
    oriimg = image_rgb.copy()
    bwimg = image.copy()
    for i in range(1,labels.shape[0]-1):
        for j in range(1,labels.shape[1]-1):
            rect = labels[i-1:i+2, j-1:j+2]
            #print(np.mean(rect))
            if labels[i, j] != np.mean(rect):#算数平均值。i,j点的像素值与走位区域的平均值不相等，则是边缘
                oriimg[i, j] = [255, 0, 0]
                bwimg[i, j] = 255
            else:
                bwimg[i, j] = 0
    #用于灰度扩张
    bwimg_wide = ndi.grey_dilation(bwimg, size=(5,5))
    # bwimg_wide = ndi.grey_dilation(bwimg, footprint=ndi.generate_binary_structure(2,1))
    # bwimg_wide = ndi.grey_dilation(bwimg_wide, footprint=ndi.generate_binary_structure(2, 1))

    # 输出
    cv2.imwrite('data/'+type_trainortest+'/water_gradient10_kuo3/'+str(index)+'.png', oriimg)
    cv2.imwrite('data/'+type_trainortest+'/wateredge_gradient10_kuo3/'+str(index)+'.png', bwimg)
    cv2.imwrite('data/'+type_trainortest+'/wateredgewide_gradient10_kuo3/' + str(index) + '.png', bwimg_wide)

    ax_oriimg = plt.subplot(2, 2, 3)
    ax_oriimg.set_title('original')
    ax_oriimg.imshow(image_rgb)
    ax_segimg = plt.subplot(2, 2, 4)
    ax_segimg.set_title('edge')
    ax_segimg.imshow(oriimg)
    #plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    axes = axes.ravel()
    ax0, ax1, ax2, ax3 = axes

    ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax0.set_title("Original")
    ax1.imshow(image_labels, cmap=plt.cm.gray, interpolation='nearest')
    ax1.set_title("Labels")
    ax2.imshow(gradient, cmap=plt.cm.gray, interpolation='nearest')
    ax2.set_title("Gradient")
    ax3.imshow(bwimg_wide, cmap=plt.cm.gray, interpolation='nearest')
    ax3.set_title("WideEdge")

    for ax in axes:
        ax.axis('off')

    fig.tight_layout()

    # plt.show()


if __name__ == '__main__':
    train_or_test = 'test'
    for index in range(30):
        print('---------',str(index),'---------')
        image = cv2.imread('data/'+train_or_test+'/image/'+str(index)+'.tif', 0)
        image_rgb = cv2.imread('data/'+train_or_test+'/image/'+str(index)+'.tif')
        image_labels = cv2.imread('data/'+train_or_test+'/label/'+str(index)+'.tif')
        if train_or_test == 'test':
            image_labels = np.zeros(image.shape, dtype=int)
        water(image, image_rgb, image_labels, index, train_or_test)

