import cv2
import numpy as np
from path import Path
import os
from tqdm import tqdm


def preprocess(root, train_list='pre_file.txt', step=4):
    root = Path(root)
    objs = [root / folder.split('/')[0] for folder in open(train_list)]

    for j, obj in enumerate(objs):
        save_path = str(obj) + '/OptFlow'.format(step)
        os.mkdir(save_path)
        frames = obj / 'Frames'
        imgs = sorted(frames.files('*.jpg'))

        for i in tqdm(range(0, len(imgs) - step, step)):
            print(str(imgs[i]), str(imgs[i+step]))
            img1 = cv2.imread(str(imgs[i]))
            img2 = cv2.imread(str(imgs[i+step]))
            mask = np.zeros_like(img1)
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
            flow_xy = TVL1.calc(img1_gray, img2_gray, None)

            mag, ang = cv2.cartToPolar(flow_xy[:, :, 0], flow_xy[:, :, 1])
            mask[:, :, 0] = ang * 180 / np.pi / 2
            mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            flow = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            cv2.imwrite(save_path+'/{}.jpg'.format(i), flow)

        print('=== {}/{} objects have been preprocessed ==='.format(j+1, len(objs)))


if __name__ == '__main__':
    preprocess('./data', step=4)
