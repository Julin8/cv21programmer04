import cv2
from average_light import aver


def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'

    return hash_str


def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def cal(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    # img1 = aver(img1)
    # img2 = aver(img2)
    hash1 = dHash(img1)
    hash2 = dHash(img2)
    return cmpHash(hash1, hash2)
