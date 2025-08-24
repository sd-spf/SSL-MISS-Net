import glob

from torch.utils import data
from torchvision import transforms
import os
import torch
import pandas as pd
import numpy as np
import random
import pickle
from PIL import Image

import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right
def get_crop_slice(target_size, dim):
    # dim is the ori shape

    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)


def pad_or_crop_image(image, target_size=(128, 144, 144)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]

    image = image[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for idx, to_pad in enumerate(todos):
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    return image

class Brain(data.Dataset):
    def __init__(self, data_file, w2v_path, transform=None, dataset_name = None):

        self.transform = transform

        self.data_file = data_file

        df = pd.read_excel(data_file,sheet_name=dataset_name)
        a = df.to_numpy()  # (506, 17)
        samples_patients = []
        samples_t1c_path = []
        samples_flair_path = []
        samples_idh_label = []
        samples_pq_label = []
        samples_tert_label = []
        samples_id_label = []
        for j in a:
            patient = j[0]
            samples_patients.append(patient)
            T1C_path = str(j[1])

            Flair_path = str(j[2])


            samples_flair_path.append(Flair_path)
            samples_t1c_path.append(T1C_path)

            Idh_label = int(j[3])

            samples_idh_label.append(Idh_label)

            pq_label = int(j[4])

            samples_pq_label.append(pq_label)

            tert_label = int(j[6])

            samples_tert_label.append(tert_label)

            id = int(j[7])
            samples_id_label.append(id)

        data_1 = np.array([samples_patients, samples_t1c_path,samples_flair_path,samples_idh_label, samples_pq_label, samples_tert_label,samples_id_label]).T

        self.data = data_1
        with open(w2v_path, 'rb') as fp:
            self.gcn_inp = np.array(pickle.load(fp), dtype=float)

    def __getitem__(self, idex):
        mri_data = self.data[idex]
        fl_img = torch.zeros((3, 256, 256))
        t1c_img = torch.zeros((3, 256, 256))
        name = mri_data[0]
        T1C_path1 = mri_data[2]

        Flair_path1 = mri_data[1]
        Flair_concatenated_result = []
        T1C_concatenated_result = []
        Flair_paths = []
        t1c_paths = []

        if str(T1C_path1) == 'nan':
            Flair_image_paths = glob.glob(Flair_path1+'/*.png')
            n1 = len(Flair_image_paths)
            for Flair_image_path in Flair_image_paths:
                Flair_paths.append(Flair_image_path)

                Flair_image = Image.open(Flair_image_path).convert('RGB')
                Flair_image = self.transform(Flair_image)
                Flair_concatenated_result.append(Flair_image)
            Flair_concatenated_result = torch.stack(Flair_concatenated_result)
            for n in range(n1):
                T1C_concatenated_result.append(t1c_img)
            T1C_concatenated_result = torch.stack(T1C_concatenated_result)

            m_d = 1

        elif str(Flair_path1) == 'nan':
            T1C_image_paths = glob.glob(T1C_path1 + '/*.png')
            n2 = len(T1C_image_paths)
            for T1C_image_path in T1C_image_paths:
                t1c_paths.append(T1C_image_path)

                T1C_image = Image.open(T1C_image_path).convert('RGB')
                T1C_image = self.transform(T1C_image)
                T1C_concatenated_result.append(T1C_image)
            T1C_concatenated_result = torch.stack(T1C_concatenated_result)
            for n in range(n2):
                Flair_concatenated_result.append(fl_img)
            Flair_concatenated_result = torch.stack(Flair_concatenated_result)

            m_d = 0

        else:
            Flair_image_paths = glob.glob(Flair_path1 + '/*.png')
            T1C_image_paths = glob.glob(T1C_path1 + '/*.png')
            n1 = len(Flair_image_paths)
            n2 = len(T1C_image_paths)
            n = min(n1,n2)
            for Flair_image_path in Flair_image_paths[:n]:
                Flair_paths.append(Flair_image_path)

                Flair_image = Image.open(Flair_image_path).convert('RGB')
                Flair_image = self.transform(Flair_image)
                Flair_concatenated_result.append(Flair_image)
            Flair_concatenated_result=torch.stack(Flair_concatenated_result)


            for T1C_image_path in T1C_image_paths[:n]:
                t1c_paths.append(T1C_image_path)

                T1C_image = Image.open(T1C_image_path).convert('RGB')
                T1C_image = self.transform(T1C_image)
                T1C_concatenated_result.append(T1C_image)
            T1C_concatenated_result = torch.stack(T1C_concatenated_result)
            m_d = -1



        label = np.zeros([7])  # list数组【0，0，，，，0】  长度6

        if int(mri_data[3]) == 1:
            label[1] = 1  #
        elif int(mri_data[3]) == 0:
            label[0] = 1  #
        else:
            pass

        if int(mri_data[4]) == 1:
            label[3] = 1
        elif int(mri_data[4]) == 0:
            label[2] = 1
        else:
            pass

        if int(mri_data[5]) == 1:
            label[5] = 1
        elif int(mri_data[5]) == 0:
            label[4] = 1
        elif int(mri_data[5]) == 2:
            label[6] = 1
        else:
            pass

        label = torch.tensor(label, dtype=torch.long)

        return T1C_concatenated_result, Flair_concatenated_result, label,self.gcn_inp,m_d,name, Flair_paths,t1c_paths

    def __len__(self):
        return len(self.data)

def get_loaders(data_files,w2v_path = None,test_dataset_name=None):

    val_data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    datasets =Brain(data_files['test'], transform=val_data_transform,w2v_path=w2v_path, dataset_name=test_dataset_name)

    return datasets

