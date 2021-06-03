import os
import cv2
from PIL import Image
from randomResult import get_random_num
from similarityCalculate import cal


def face_detect(image_path):
    img = cv2.imread(image_path)
    facedetect = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    # 转换成灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 人脸检测，距离，最小检测像素点数，最小像素点大小
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    box = [0, 0, 0, 0]

    for (x, y, w, h) in faces:
        box[0] = x
        box[1] = y
        box[2] = x + w
        box[3] = y + h
    if box != [0, 0, 0, 0]:
        return None
    else:
        return box


f = open('result.txt', 'w')
for root, dirs, files in os.walk('test_adjust_face'):
    for file in files:
        path1 = os.path.join(root, file)
        box = face_detect(path1)
        img = Image.open(path1)
        if box:
            img = img.crop(box)

        # 先随机分配一个label
        label = get_random_num()
        n = cal(path1, "gallery_adjust_no_face/" + label + ".jpg")
        # n = cal_by_path(path1, "gallery_face/" + label + ".jpg") + cal_by_path('val_nose/' + file, 'gallery_nose/'
        # + label + '.jpg')

        for froot, fdirs, ffiles in os.walk('gallery_adjust_no_face'):
            for ffile in ffiles:
                path2 = os.path.join(froot, ffile)

                # if cal_by_path(path1, path2) + cal_by_path('val_nose/' + file, 'gallery_nose/' + ffile) <= n \
                #        and cal_by_path(path1, path2) != -1:

                if cal(path1, path2) <= n and cal(path1, path2) != -1:
                    n = cal(path1, path2)
                    label = ffile.replace(".jpg", "")

        print("*******************************")
        print(file + ' ' + label)
        f.writelines(file + " " + label + '\n')
