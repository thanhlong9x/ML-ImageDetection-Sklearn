import os
import cv2
import numpy as np
from sklearn import svm

import matplotlib.pyplot as plt

test_path = '/home/mrtu/Downloads/Misa/ImageDetection/src/test'
training = '/home/mrtu/Downloads/Misa/ImageDetection/src/training'


def show_image(image):
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


class Mnist:

    def __init__(self):
        self.model = None

    def get_training_data(self):

        x_train = []
        y_train = []
        for v in os.listdir(training):
            if not v.isdigit():
                continue
            target = int(v)

            images_path = os.path.join(training, v)
            for img_path in os.listdir(images_path):
                real_path = os.path.join(images_path, img_path)
                img = cv2.imread(real_path, 0)

                image = self.preprocess(img)
                x_train.append(self.get_feature(image))
                y_train.append(target)

        return np.array(x_train), np.array(y_train)

    def train(self):
        self.build_model()

        x_train, y_train = self.get_training_data()

        x_train = np.asarray(x_train, dtype='float')

        self.model.fit(x_train, y_train)
        accuracy = self.model.score(x_train, y_train)
        print("accuracy train: %f" % accuracy)
        return accuracy

    def test(self):
        count_correct = 0
        count = 0
        for v in os.listdir(test_path):
            if not v.isdigit():
                continue
            target = int(v)

            images_path = os.path.join(test_path, v)
            for img_path in os.listdir(images_path):
                count += 1
                real_path = os.path.join(images_path, img_path)
                img = cv2.imread(real_path, 0)
                # print(real_path)
                result = self.predict(img)
                if result == target:
                    count_correct += 1
        print('accuracy test: {}'.format(float(count_correct / count)))

    def preprocess(self, image):
        image = 255 - image
        # show_image(image)
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        [x, y, w, h] = cv2.boundingRect(contours[0])
        image = image[y:y + h, x:x + w]
        image = cv2.resize(image, (28, 28))
        return image

    def get_feature(self, img):
        # Trả về một vector đặc trưng
        # show_image(img)
        img = np.asarray(img, dtype='float')
        return img.ravel()

    def build_model(self):
        self.model = svm.SVC(C = 1, kernel='rbf')

    def predict(self, image):
        image = self.preprocess(image)
        feature = np.array([self.get_feature(image)])
        feature = feature.astype(float)
        return int(self.model.predict(feature)[0])


if __name__ == "__main__":
    obj = Mnist()
    obj.train()
    obj.test()
