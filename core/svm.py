import os
import cv2
import math
import time
import numpy as np
import tqdm
from skimage.feature import hog
from sklearn.svm import LinearSVC
from skimage import feature as skif


class Classifier(object):
    def __init__(self, filePath):
        self.filePath = filePath

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_data(self):
        TrainData = []
        TestData = []
        for childDir in os.listdir(self.filePath):
            if 'data_batch' in childDir:
                f = os.path.join(self.filePath, childDir)
                data = self.unpickle(f)
                # train = np.reshape(data[str.encode('data')], (10000, 3, 32 * 32))
                # If your python version do not support to use this way to transport str to bytes.
                # Think another way and you can.
                train = np.reshape(data[b'data'], (1000, 3, 32 * 32))
                labels = np.reshape(data[b'labels'], (1000, 1))
                fileNames = np.reshape(data[b'filenames'], (1000, 1))
                datalebels = zip(train, labels, fileNames)
                TrainData.extend(datalebels)
            if childDir == "test_batch":
                f = os.path.join(self.filePath, childDir)
                data = self.unpickle(f)
                test = np.reshape(data[b'data'], (1000, 3, 32 * 32))
                labels = np.reshape(data[b'labels'], (1000, 1))
                fileNames = np.reshape(data[b'filenames'], (1000, 1))
                TestData.extend(zip(test, labels, fileNames))
        print("data read finished!")
        return TrainData, TestData

    def get_hog_feat(self, image, stride=8, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        cx, cy = pixels_per_cell
        bx, by = cells_per_block
        sx, sy = image.shape
        n_cellsx = int(np.floor(sx // cx))  # number of cells in x
        n_cellsy = int(np.floor(sy // cy))  # number of cells in y
        n_blocksx = (n_cellsx - bx) + 1
        n_blocksy = (n_cellsy - by) + 1
        gx = np.zeros((sx, sy), dtype=np.float32)
        gy = np.zeros((sx, sy), dtype=np.float32)
        eps = 1e-5
        grad = np.zeros((sx, sy, 2), dtype=np.float32)
        for i in range(1, sx-1):
            for j in range(1, sy-1):
                gx[i, j] = image[i, j-1] - image[i, j+1]
                gy[i, j] = image[i+1, j] - image[i-1, j]
                grad[i, j, 0] = np.arctan(gy[i, j] / (gx[i, j] + eps)) * 180 / math.pi
                if gx[i, j] < 0:
                    grad[i, j, 0] += 180
                grad[i, j, 0] = (grad[i, j, 0] + 360) % 360
                grad[i, j, 1] = np.sqrt(gy[i, j] ** 2 + gx[i, j] ** 2)
        normalised_blocks = np.zeros((n_blocksy, n_blocksx, by * bx * orientations))
        for y in range(n_blocksy):
            for x in range(n_blocksx):
                block = grad[y*stride:y*stride+16, x*stride:x*stride+16]
                hist_block = np.zeros(32, dtype=np.float32)
                eps = 1e-5
                for k in range(by):
                    for m in range(bx):
                        cell = block[k*8:(k+1)*8, m*8:(m+1)*8]
                        hist_cell = np.zeros(8, dtype=np.float32)
                        for i in range(cy):
                            for j in range(cx):
                                n = int(cell[i, j, 0] / 45)
                                hist_cell[n] += cell[i, j, 1]
                        hist_block[(k * bx + m) * orientations:(k * bx + m + 1) * orientations] = hist_cell[:]
                normalised_blocks[y, x, :] = hist_block / np.sqrt(hist_block.sum() ** 2 + eps)
        return normalised_blocks.ravel()

    def get_lbp_data(self, images_data, hist_size=256, lbp_radius=1, lbp_point=8):
        n_images = images_data.shape[0]
        hist = np.zeros((n_images, hist_size))
        for i in np.arange(n_images):
            # 使用LBP方法提取图像的纹理特征.
            lbp = skif.local_binary_pattern(images_data[i], lbp_point, lbp_radius, 'default')
            # 统计图像的直方图
            max_bins = int(lbp.max() + 1)
            # hist size:256
            hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))

        return hist

    def get_feat(self, TrainData, TestData):
        train_feat = []
        test_feat = []
        for data in tqdm.tqdm(TestData):
            image = np.reshape(data[0].T, (32, 32, 3))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.
            fd = self.get_hog_feat(gray)
            # fd = hog(gray, 9, [8, 8], [2, 2])
            fd = np.concatenate((fd, data[1]))
            test_feat.append(fd)
        test_feat = np.array(test_feat)
        print("Test features are extracted and saved.")
        for data in tqdm.tqdm(TrainData):
            image = np.reshape(data[0].T, (32, 32, 3))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
            fd = self.get_hog_feat(gray)
            # fd = hog(gray, 9, [8, 8], [2, 2])
            fd = np.concatenate((fd, data[1]))
            train_feat.append(fd)
        train_feat = np.array(train_feat)
        print("Train features are extracted and saved.")
        return train_feat, test_feat

    def classification(self, train_feat, test_feat):
        t0 = time.time()
        clf = LinearSVC()
        print("Training a Linear SVM Classifier.")
        clf.fit(train_feat[:, :-1], train_feat[:, -1])
        predict_result = clf.predict(test_feat[:, :-1])
        num = 0
        for i in range(len(predict_result)):
            if int(predict_result[i]) == int(test_feat[i, -1]):
                num += 1
        rate = float(num) / len(predict_result)
        t1 = time.time()
        print('The classification accuracy is %f' % rate)
        print('The cast of time is :%f' % (t1 - t0))

    def run(self):
        TrainData, TestData = self.get_data()
        train_feat, test_feat = self.get_feat(TrainData, TestData)
        self.classification(train_feat, test_feat)


if __name__ == '__main__':
    filePath = './data/train/'
    cf = Classifier(filePath)
    cf.run()