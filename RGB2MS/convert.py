from model.Unet import UNet
import scipy.io
from dataloader import *

def image_open(filepath):
    I = Image.open(filepath)
    I = np.array(I,dtype='float32').transpose(2, 0, 1)/255.0
    output = torch.tensor(I).unsqueeze(dim=0)
    return output
def save_mat(filepath,image_name, out):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    out = out * 255.0
    out = out.squeeze()
    out = rearrange(out,'c h w -> h w c')
    out = np.array(out).astype('double')
    scipy.io.savemat("{}/{}".format(filepath, image_name[:-4] + '.mat'), {"mat_data": out},do_compression=True)
    print("{}/{}".format(filepath, image_name[:-4] + '.mat'))
    return out

# cuda
iscuda = True
device = torch.device("cuda:0" if iscuda else "cpu")
print(device)

# save path, img path, model path
SAVE_PATH = "dataset/nus_dataset/MS_8ch/"
im_dir = "dataset/nus_dataset/RGB/"
MODEL_PATH = 'weights/RGB2MS/Unet.pth'
image_filenames = [x for x in listdir(im_dir) if is_image_file(x)]

# model
net = UNet(in_ch=3, out_ch=8, bilinear=False).to(device)
net = torch.nn.DataParallel(net)

# load weight
if os.path.isfile(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    print('successfully loaded checkpoints!')
else:
    print("check MODEL_PATH")
    assert os.path.isfile(MODEL_PATH) == True

net.eval()
with torch.no_grad():
    for iteration,image_name in enumerate(image_filenames):
        rgb = image_open(im_dir + image_name).to(device)
        out = net(rgb)
        out = out.detach().squeeze(0).cpu()
        out1 = out

        # save mat
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        save_mat(SAVE_PATH + '/mat/', image_name.split('.')[0]+'.'+image_name.split('.')[1], out1)





