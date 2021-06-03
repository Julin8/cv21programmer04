import cv2
from PIL import Image
import os


def dataProcess(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            image_path = os.path.join(root, file)
            image = Image.open(image_path)
            img = cv2.imread(image_path)
            facedetect = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
            # 转换成灰度图像
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 人脸检测，距离，最小检测像素点数，最小像素点大小
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            box = [0, 0, img.shape[0], img.shape[1]]
            for (x, y, w, h) in faces:
                # 画矩形，添加英文标识
                box[0] = x
                box[1] = y
                box[2] = x+w
                box[3] = y+h
                # box.append(x)
                # box.append(y)
                # box.append(x+w)
                # box.append(y+h)
            im = image.crop(box)
            im.save(output_dir + '/' + file)


def dataProcess_nose(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            image_path = os.path.join(root, file)
            image = Image.open(image_path)
            img = cv2.imread(image_path)
            nose_detect = cv2.CascadeClassifier('./haarcascade_mcs_nose.xml')
            # 转换成灰度图像
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 人脸检测，距离，最小检测像素点数，最小像素点大小
            noses = nose_detect.detectMultiScale(gray, 1.3, 5)
            box = [0, 0, img.shape[0], img.shape[1]]
            for (x, y, w, h) in noses:
                # 画矩形，添加英文标识
                box[0] = x
                box[1] = y
                box[2] = x+w
                box[3] = y+h
            im = image.crop(box)
            im.save(output_dir + '/' + file)


# dataProcess("gallery", "gallery_face")
# dataProcess("val", "val_face")
dataProcess('val_adjust_no', 'val_adjust_no_face')
dataProcess('test_adjust', 'test_adjust_face')
