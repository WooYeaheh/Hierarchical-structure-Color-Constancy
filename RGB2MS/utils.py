import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import os


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", 'tiff', 'bmp', '.mat'])


def load_img(filepath):
    img = Image.open(filepath)  # .convert('RGB')
    # img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0  # 0 ~ 255 사이로 변환
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)  # 0 이하의 값은 0으로, 255 이상의 값은 255로
    image_numpy = image_numpy.astype(np.uint8)  # 정수 값으로 변환
    image_pil = image_numpy
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def save_np_img(image_numpy, filename):
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def record_loss(loss_csv, epoch, loss0=0, loss1=0, loss2=0, loss3=0, loss4=0, loss5=0, loss6=0, loss7=0, loss8=0):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{},{},{},{},{}\n'.format(epoch, loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8))

    loss_csv.flush()
    loss_csv.close


def record_loss2(loss_csv, epoch, train_loss, loss2, loss3):
    """ Record many results."""
    loss_csv.write('{},{},{},{}\n'.format(epoch, train_loss, loss2, loss3))

    loss_csv.flush()
    loss_csv.close


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.unsqueeze(2)
    return gray


def rgb2gray_tensor(rgb):
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.unsqueeze(1)
    return gray


def save_gray_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    # print(image_numpy)
    image_numpy = image_numpy * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def get_reconstruction(input, dimension, model, num_split=1):
    """As the limited GPU memory split the input."""
    print(input.shape[3])
    input_split = torch.split(input, int(input.shape[3] / num_split), dim=dimension)
    output_split = []
    for i in range(num_split):
        var_input = Variable(input_split[i].cuda(), volatile=True)
        var_output = model(var_input)
        # var_output = model(var_input)
        output_split.append(var_output.data)
        if i == 0:
            output = output_split[i]
        else:
            output = torch.cat((output, output_split[i]), dim=dimension)

    return output