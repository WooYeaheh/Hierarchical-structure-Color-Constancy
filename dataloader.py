import os
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
import random
import scipy
from scipy import io
import mat73
from einops import rearrange

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".bmp"])

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

def is_txt_file(filename):
    return any(filename.endswith(extension) for extension in [".txt"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def get_patch_2tensor(img_in,wb_img_in, patch_size=256): # img_in : 이미지 파일 w,h | wb_img_in : 텐서 c,h,w
    c ,ih, iw = img_in.size()
    ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    img_in = img_in[:,iy:iy+ip,ix:ix+ip]
    wb_img_in = wb_img_in[:,iy:iy+ip,ix:ix+ip]

    return img_in,wb_img_in

class DatasetFromFolder_NUS_fold(data.Dataset):
    def __init__(self, data_dir, isTrain, fold_num = 0, transform=None):
        super(DatasetFromFolder_NUS_fold, self).__init__()

        self.Train = isTrain
        self.folds_num = fold_num
        self.fold_path = os.path.join(data_dir, 'fold_nus.mat')
        folds = mat73.loadmat(self.fold_path)
        img_idx = folds["tr_split" if self.Train else "te_split"][self.folds_num]

        self.im_dir = os.path.join(data_dir, 'rgb/')
        self.gt_label_dir = os.path.join(data_dir, 'label/')
        self.MS_dir = os.path.join(data_dir,'vis_8ch/mat/')

        self.image_filenames = [join(self.im_dir, x) for x in listdir(self.im_dir) if is_image_file(x)]
        self.gt_label_filenames = [join(self.gt_label_dir, x) for x in listdir(self.gt_label_dir) if is_txt_file(x)]
        self.MS_filenames = [join(self.MS_dir,x) for x in listdir(self.MS_dir) if is_mat_file(x)]

        self.image_filenames = [self.image_filenames[i-1] for i in img_idx]
        self.gt_label_filenames = [self.gt_label_filenames[i-1] for i in img_idx]
        self.MS_filenames = [self.MS_filenames[i - 1] for i in img_idx]

        self.transform = transform

        print(self.image_filenames)
        print(self.MS_filenames)
        print('총 이미지 개수: {}'.format(len(self.image_filenames)))
    def __getitem__(self, index):

        input_im = load_img(self.image_filenames[index])
        input_im = self.transform(input_im).float()

        MS_im = scipy.io.loadmat(self.MS_filenames[index])
        MS_im = MS_im['mat_data']
        MS_im = self.transform(MS_im).float()

        gt_label = np.zeros(shape=(3))

        with open(self.gt_label_filenames[index]) as f:
            lines = f.readlines()[0:3]
        for i in range(3):
            gt_label[i] = lines[i]

        gt_label = torch.from_numpy(gt_label)

        if self.Train == True:
            input_full, MS_full = get_patch_2tensor(input_im, MS_im, patch_size=256)
            input_area, MS_area = get_patch_2tensor(input_full, MS_full, patch_size=128)
            input_patch, MS_patch = get_patch_2tensor(input_area, MS_area, patch_size=64)
            return input_full, input_area, input_patch, MS_full, MS_area, MS_patch, gt_label
        else:
            return input_im, MS_im, gt_label

    def __len__(self):
        return len(self.image_filenames)


class DatasetFromFolder_MS(data.Dataset):
    def __init__(self, image_dir, patch_size, isTrain, transform=None):
        super(DatasetFromFolder_MS, self).__init__()
        self.Train = isTrain

        self.im_dir = os.path.join(image_dir,  'RGB/')
        self.MS_dir = os.path.join(image_dir,  'MS_ICVL/')

        self.image_filenames = [join(self.im_dir, x) for x in listdir(self.im_dir) if is_image_file(x)]
        self.MS_image_filenames = [join(self.MS_dir, x) for x in listdir(self.MS_dir) if is_mat_file(x)]
        self.patch_size = patch_size
        self.transform = transform

        print(self.image_filenames)
        print(self.MS_image_filenames)
    def __getitem__(self, index):
        input_im = load_img(self.image_filenames[index])
        input_im = self.transform(input_im)
        MS_im = scipy.io.loadmat(self.MS_image_filenames[index])

        MS_im = rearrange(torch.tensor((MS_im['mat_data'])),'w h c -> c h w')/255.0 # .flip(0,1).rot90(dims=(1,2),k=-1)#.flip(2,1) # np.asarray
        (channel,new_height,new_width) = MS_im.shape
        input_im = input_im.resize((new_width, new_height))

        if self.Train == True:
            input_patch, MS_patch = get_patch_2tensor(input_im, MS_im, self.patch_size)
            return input_patch, MS_im
        else:
            return input_im, MS_im

    def __len__(self):
        return len(self.image_filenames)
