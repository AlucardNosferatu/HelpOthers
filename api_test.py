import SimpleITK
import cv2
import numpy as np


def read_nrrd(img, mask):
    img_array = SimpleITK.GetArrayFromImage(img)
    mask_array = SimpleITK.GetArrayFromImage(mask)
    img_array = img_array.astype('float32')
    mask_array = mask_array.astype('float32')
    for i in range(img_array.shape[0]):
        img_array[i, :, :] = img_array[i, :, :] / np.max(img_array[i, :, :])
        img_array_slice = img_array[i, :, :]
        mask_array_slice = np.copy(mask_array[i, :, :])
        cv2.imshow('img', img_array_slice)
        cv2.imshow('mask', mask_array_slice)
        cv2.waitKey()


if __name__ == '__main__':
    img_path = SimpleITK.ReadImage('data/featureExtraction/brain1/brain1_image.nrrd')
    mask_path = SimpleITK.ReadImage('data/featureExtraction/brain1/brain1_label.nrrd')
    read_nrrd(img_path, mask_path)
    print('Done')
