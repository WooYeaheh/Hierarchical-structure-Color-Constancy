import easydict
import torch
import os
import scipy.io
import scipy.io
import numpy as np
from PIL import Image
import mat73
from einops import rearrange
def save_mat(filepath,image_name, out):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    out = out * 255.0
    out = out.squeeze()
    out = rearrange(out,'c h w -> h w c')
    out = np.array(out).astype('uint8')
    scipy.io.savemat("{}/{}".format(filepath, image_name[:-4] + '.mat'), {"mat_data": out})
    return out
def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])
def image_open(filepath):
    I = Image.open(filepath)
    I = np.array(I,dtype='float32')
    if len(I.shape) ==2: # gray
        I = I.transpose(0, 1)/255.0
        output = torch.tensor(I).unsqueeze(dim=0).unsqueeze(dim=0)
    else:
        I = I.transpose(2, 0, 1) / 255.0
        output = torch.tensor(I).unsqueeze(dim=0)

    return output
def tensor_to_image(tensor):
    if len(tensor.shape)==4 : #batch 포함
        b,c,h,w = tensor.size()
        tensor = tensor[0]
        if c == 3:
            pass
        elif c == 1:
            tensor = tensor.repeat(3,1,1)
        output = np.transpose(np.array(tensor), (2,1,0))
        output = Image.fromarray(np.clip(output * 255.0, 0, 255.0).astype('uint8'))
    elif len(tensor.shape) == 3:
        c,h,w = tensor.size()
        if c == 3:
            pass
        elif c == 1:
            tensor = tensor.repeat(3, 1, 1)
        output = np.transpose(np.array(tensor), (2,1,0))
        output = Image.fromarray(np.clip(output * 255.0, 0, 255.0).astype('uint8'))
    elif len(tensor.shape) == 2:
        h,w = tensor.size()
        tensor = tensor.repeat(3,1,1)
        output = np.transpose(np.array(tensor), (2, 1, 0))
        output = Image.fromarray(np.clip(output * 255.0, 0, 255.0).astype('uint8'))
    return output
def matsave(filepath,file):
    return scipy.io.savemat(filepath, {"mat_data": file})
def np_to_image(tensor):
    tensor = tensor.squeeze().repeat(1,1,1)
    #print(tensor.size())
    output = np.transpose(np.array(tensor), (1,2, 0))
    output = Image.fromarray(np.clip(output * 255.0, 0, 255.0).astype('uint8'))
    return output

IMAGE_DIR = 'dataset/ICVL_dataset/HSI/'
SAVE_PATH = 'dataset/ICVL_dataset/MS_ICVL/'

image_filenames = [x for x in os.listdir(IMAGE_DIR)]

from einops import rearrange

for index,image_name in enumerate(image_filenames):
    if is_mat_file(image_name):
        data = mat73.loadmat(IMAGE_DIR + image_name)
    else: pass
    matdata = data['rad']
    matdata = torch.tensor(matdata)
    matdata = rearrange(matdata,'w h c -> c h w')
    if max < matdata.max():
        max = matdata.max()

    '''Seperate MS 8ch'''
    mat = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(4):
        mat[0] += matdata[i].unsqueeze(0)
    for i in range(4, 8):
        mat[1] += matdata[i].unsqueeze(0)
    for i in range(8, 12):
        mat[2] += matdata[i].unsqueeze(0)
    for i in range(12, 16):
        mat[3] += matdata[i].unsqueeze(0)
    for i in range(16, 20):
        mat[4] += matdata[i].unsqueeze(0)
    for i in range(20, 24):
        mat[5] += matdata[i].unsqueeze(0)
    for i in range(24, 28):
        mat[6] += matdata[i].unsqueeze(0)
    for i in range(28, 31):
        mat[7] += matdata[i].unsqueeze(0)

    mat[0] = mat[0] / (4.0)
    mat[0] = mat[0] / 4092.0
    mat[1] = mat[1] / (4.0)
    mat[1] = mat[1] / 4092.0
    mat[2] = mat[2] / (4.0)
    mat[2] = mat[2] / 4092.0
    mat[3] = mat[3] / (4.0)
    mat[3] = mat[3] / 4092.0
    mat[4] = mat[4] / (4.0)
    mat[4] = mat[4] / 4092.0
    mat[5] = mat[5] / (4.0)
    mat[5] = mat[5] / 4092.0
    mat[6] = mat[6] / (4.0)
    mat[6] = mat[6] / 4092.0
    mat[7] = mat[7] / (3.0)
    mat[7] = mat[7] / 4092.0

    mat_img = torch.cat((mat[7],mat[6],mat[5],mat[4], mat[3], mat[2], mat[1], mat[0]), dim=0).rot90(-1, dims=(1, 2))  # .flip(dims=(2,1)).flip(dims=(0,1))
    save_mat(SAVE_PATH,image_name,mat_img)




