from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import utils_paths
import numpy as np
import random
import cv2
import os
import pywt


def WPEnergy(data, fs, wavelet, maxlevel=2):
    iter_freqs = [
        {'name': 'Delta', 'fmin': 0, 'fmax': 4},
        {'name': 'Theta', 'fmin': 4, 'fmax': 8},
        {'name': 'Alpha', 'fmin': 8, 'fmax': 13},
        {'name': 'Beta', 'fmin': 13, 'fmax': 35},]

    # 小波包分解
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # 计算maxlevel最小频段的带宽
    freqBand = fs / (2 ** maxlevel)
    # 定义能量数组
    energy = []
    # 循环遍历计算四个频段对应的能量
    for iter in range(len(iter_freqs)):
        iterEnergy = 0.0
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if (iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax):
                # 计算对应频段的累加和
                iterEnergy += pow(np.linalg.norm(wp[freqTree[i]].data, ord=None), 2)
        # 保存四个频段对应的能量和
        energy.append(iterEnergy)
    return float(energy[0]+energy[1]+energy[2]+energy[3])


def with_we_r(path, flag):
    # 拿到图像数据路径，方便后续读取
    imagePaths = sorted(list(utils_paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    data1 = []
    labels = []

    for imagePath in imagePaths:
        # 读取图像数据
        image1 = cv2.imread(imagePath)
        # image = cv2.resize(image, (256, 256))
        b, g, r = cv2.split(image1)
        b_ = pywt.wavedec2(b, 'db4', level=2)
        g_ = pywt.wavedec2(g, 'db4', level=2)
        r_ = pywt.wavedec2(r, 'db4', level=2)
        b_ = WPEnergy(b_, fs=250, wavelet='db4', maxlevel=2)
        g_ = WPEnergy(g_, fs=250, wavelet='db4', maxlevel=2)
        r_ = WPEnergy(r_, fs=250, wavelet='db4', maxlevel=2)
        if b_ >= g_:
            k = b
        else:
            k = g
        if k >= r_:
            pass
        else:
            k = r
        data1.append(k)
        # 读取标签
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    # 对图像数据做scale操作
    data = np.array(data1, dtype="float") / 255.0
    labels = np.array(labels)

    # 数据集切分
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # 转换标签为one-hot encoding格式
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    # 数据增强处理
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    if flag == 'train':
        return trainX, trainY
    else:
        return testX, testY


def with_we_h(path, flag):
    # 拿到图像数据路径，方便后续读取
    imagePaths = sorted(list(utils_paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    data2 = []
    labels = []

    for imagePath in imagePaths:
        # 读取图像数据
        image1 = cv2.imread(imagePath)
        # image = cv2.resize(image, (256, 256))
        image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image2)
        h_ = pywt.wavedec2(h, 'db4', level=2)
        s_ = pywt.wavedec2(v, 'db4', level=2)
        v_ = pywt.wavedec2(s, 'db4', level=2)
        h_ = WPEnergy(h_, fs=250, wavelet='db4', maxlevel=2)
        s_ = WPEnergy(s_, fs=250, wavelet='db4', maxlevel=2)
        v_ = WPEnergy(v_, fs=250, wavelet='db4', maxlevel=2)
        if h_ >= s_:
            k = h
        else:
            k = s
        if k >= v_:
            pass
        else:
            k = v
        data2.append(k)
        # 读取标签
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    # 对图像数据做scale操作
    data = np.array(data2, dtype="float") / 255.0
    labels = np.array(labels)

    # 数据集切分
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # 转换标签为one-hot encoding格式
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    # 数据增强处理
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    if flag == 'train':
        return trainX, trainY
    else:
        return testX, testY


def without_we(path, flag):
    # 拿到图像数据路径，方便后续读取
    imagePaths = sorted(list(utils_paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    data3 = []
    labels = []
    for imagePath in imagePaths:
        # 读取图像数据
        image1 = cv2.imread(imagePath)
        # image = cv2.resize(image, (256, 256))
        image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(image1)
        h, s, v = cv2.split(image2)
        data3.append(b)
        data3.append(g)
        data3.append(r)
        data3.append(h)
        data3.append(s)
        data3.append(v)
        label = imagePath.split(os.path.sep)[-2]
        for i in range(6):
            labels.append(label)
    # 数据集切分
    (trainX, testX, trainY, testY) = train_test_split(data3, labels, test_size=0.25, random_state=42)

    # 转换标签为one-hot encoding格式
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    # 数据增强处理
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    if flag == 'train':
        return trainX, trainY
    else:
        return testX, testY


def with_rgb(path, flag):
    # 拿到图像数据路径，方便后续读取
    imagePaths = sorted(list(utils_paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    data4 = []
    data5 = []
    data6 = []
    labels = []
    for imagePath in imagePaths:
        # 读取图像数据
        image1 = cv2.imread(imagePath)
        # image = cv2.resize(image, (256, 256))
        b, g, r = cv2.split(image1)
        data4.append(b)
        data5.append(g)
        data6.append(r)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    # 数据集切分
    (trainX_b, testX_b, trainY_b, testY_b) = train_test_split(data4, labels, test_size=0.25, random_state=42)
    (trainX_g, testX_g, trainY_g, testY_g) = train_test_split(data5, labels, test_size=0.25, random_state=42)
    (trainX_r, testX_r, trainY_r, testY_r) = train_test_split(data6, labels, test_size=0.25, random_state=42)

    # 转换标签为one-hot encoding格式
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY_b)
    testY = lb.transform(testY_b)

    # 数据增强处理
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    if flag == 'train':
        return trainX_b, trainX_g, trainX_r, trainY
    else:
        return testX_b, testX_g, testX_r, testY