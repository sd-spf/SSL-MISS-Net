import glob

from torch.utils import data
from torchvision import transforms

import torch
import pandas as pd
import numpy as np
import random
import pickle
from PIL import Image


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
    def __init__(self, data_file, w2v_path, transform=None, phase='train',dataset_name = None):
        self.transform = transform
        self.data_file = data_file
        self.phase = phase

        df = pd.read_excel(data_file,sheet_name=dataset_name)
        a = df.to_numpy()  # (506, 17)
        samples_patients = []
        samples_t1c_path = []
        samples_flair_path = []
        samples_idh_label = []
        samples_pq_label = []
        samples_tert_label = []
        samples_center_id_label = []
        for j in a:
            patient = j[0]
            T1C_path = str(j[1])

            Flair_path = str(j[2])
            if T1C_path != 'nan' and Flair_path != 'nan':
                T1C_images = glob.glob(T1C_path+'/*.png')
                flair_images = glob.glob(Flair_path+'/*.png')
                n1 = len(T1C_images)
                n2 = len(flair_images)
                if n1 != n2:

                    min_value = min(n1,n2)
                    T1C_images = T1C_images[:min_value]
                    flair_images = flair_images[:min_value]

                    samples_t1c_path += T1C_images
                    samples_flair_path += flair_images

                    patient = np.array(patient)
                    patients = list(patient.repeat(min_value, 0))
                    samples_patients += patients

                    Idh_label = np.array(int(j[3]))
                    idh_labels = list(Idh_label.repeat(min_value, 0))
                    samples_idh_label += idh_labels
                    pq_label = np.array(int(j[4]))
                    pq_labels = list(pq_label.repeat(min_value, 0))
                    samples_pq_label += pq_labels

                    tert_label = np.array(int(j[6]))
                    tert_labels = list(tert_label.repeat(min_value, 0))
                    samples_tert_label += tert_labels

                    center_id = np.array(int(j[7]))
                    center_ids = list(center_id.repeat(min_value, 0))
                    samples_center_id_label += center_ids

                else:
                    samples_t1c_path += T1C_images
                    samples_flair_path += flair_images

                    patient = np.array(patient)
                    patients = list(patient.repeat(n1, 0))
                    samples_patients += patients

                    Idh_label = np.array(int(j[3]))
                    idh_labels = list(Idh_label.repeat(n1,0))
                    samples_idh_label += idh_labels
                    pq_label = np.array(int(j[4]))
                    pq_labels = list(pq_label.repeat(n1,0))
                    samples_pq_label += pq_labels

                    tert_label = np.array(int(j[6]))
                    tert_labels = list(tert_label.repeat(n1,0))
                    samples_tert_label += tert_labels

                    center_id = np.array(int(j[7]))
                    center_ids = list(center_id.repeat(n1, 0))
                    samples_center_id_label += center_ids

            elif T1C_path == 'nan' and Flair_path != 'nan':
                flair_images = glob.glob(Flair_path + '/*.png')
                n1 = len(flair_images)
                samples_flair_path += flair_images

                T1C_path = np.array(T1C_path)
                T1C_paths = list(T1C_path.repeat(n1, 0))
                samples_t1c_path += T1C_paths

                patient = np.array(patient)
                patients = list(patient.repeat(n1, 0))
                samples_patients += patients

                Idh_label = np.array(int(j[3]))
                idh_labels = list(Idh_label.repeat(n1, 0))
                samples_idh_label += idh_labels
                pq_label = np.array(int(j[4]))
                pq_labels = list(pq_label.repeat(n1, 0))
                samples_pq_label += pq_labels

                tert_label = np.array(int(j[6]))
                tert_labels = list(tert_label.repeat(n1, 0))
                samples_tert_label += tert_labels

                center_id = np.array(int(j[7]))
                center_ids = list(center_id.repeat(n1, 0))
                samples_center_id_label += center_ids
            else:
                T1C_images = glob.glob(T1C_path + '/*.png')
                n1 = len(T1C_images)
                samples_t1c_path += T1C_images

                Flair_path = np.array(Flair_path)
                Flair_paths = list(Flair_path.repeat(n1, 0))
                samples_flair_path += Flair_paths

                patient = np.array(patient)
                patients = list(patient.repeat(n1, 0))
                samples_patients += patients

                Idh_label = np.array(int(j[3]))
                idh_labels = list(Idh_label.repeat(n1, 0))
                samples_idh_label += idh_labels
                pq_label = np.array(int(j[4]))
                pq_labels = list(pq_label.repeat(n1, 0))
                samples_pq_label += pq_labels

                tert_label = np.array(int(j[6]))
                tert_labels = list(tert_label.repeat(n1, 0))
                samples_tert_label += tert_labels

                center_id = np.array(int(j[7]))
                center_ids = list(center_id.repeat(n1, 0))
                samples_center_id_label += center_ids
        data_1 = list(zip(samples_patients, samples_t1c_path,samples_flair_path,samples_idh_label, samples_pq_label, samples_tert_label,samples_center_id_label))
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

        if str(T1C_path1) == 'nan':
            Flair_image = Image.open(Flair_path1).convert('RGB')
            Flair_image = self.transform(Flair_image)
            T1C_image = t1c_img
            m_d = 1

        elif str(Flair_path1) == 'nan':
            T1C_image = Image.open(T1C_path1).convert('RGB')
            T1C_image = self.transform(T1C_image)
            Flair_image = fl_img
            m_d = 0

        else:
            T1C_image = Image.open(T1C_path1).convert('RGB')
            Flair_image = Image.open(Flair_path1).convert('RGB')
            T1C_image = self.transform(T1C_image)
            Flair_image = self.transform(Flair_image)
            m_d = -1

        label = np.zeros([7])

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

        return T1C_image, Flair_image, label,self.gcn_inp,m_d,name

    def __len__(self):
        return len(self.data)

def get_loaders(self, data_files, w2v_path = None):
    rs = np.random.RandomState(1234)
    train_data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90, expand=False),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    datasets = Brain(data_files['train'],transform=train_data_transform, phase='train',w2v_path = w2v_path, dataset_name = self.train_sheet_name)

    return datasets

