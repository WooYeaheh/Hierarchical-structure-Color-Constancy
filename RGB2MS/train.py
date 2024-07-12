import argparse
import torch.backends.cudnn as cudnn
from torchvision.transforms import *
from dataloader import *
from torch.utils.data import DataLoader
from model.Unet import UNet as UNetKD
import torch.nn as nn
import torch.optim as optim
from utils import *
import vgg16_loss

# Training settings
parser = argparse.ArgumentParser(description='UNET for RGB2MS Conversion')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=1000, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_dir', default='dataset/ICVL_dataset/', type=str, help='Dataset dir')
parser.add_argument('--save_folder', default='weights/RGB2MS/', type=str, help='save path')
parser.add_argument('--patch_size', type=int, default=256, help='patch size in training phase')

opt = parser.parse_args()

SAVE_FOLDER = opt.save_folder
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)

if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

def transform():
    return Compose([
        ToTensor(),
    ])

print('===> Loading datasets')
train_data = DatasetFromFolder_MS(image_dir=opt.data_dir,patch_size=opt.patch_size,isTrain=True,transform=transform())
dataloader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)

device = "cuda"

model = UNetKD(in_ch=3, out_ch=8, bilinear=False).cuda()
model = torch.nn.DataParallel(model)


L1_loss = nn.L1Loss().to(device)
VGG_loss = vgg16_loss.VGG16Loss().to(device)
params = nn.ParameterList(list(model.parameters()))
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=20,threshold=0.1,verbose=True)

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

net_path = SAVE_FOLDER + "/net_epoch_{}.pth".format(opt.epoch_count)
loss_csv = open(SAVE_FOLDER + '/loss_.csv', 'w+')
loss_csv.write('{},{}\n'.format('epoch', 'total loss'))

if os.path.isfile(net_path):
    print('Loading pre-trained network!')
    checkpoint = torch.load(net_path)
    model.load_state_dict(checkpoint['model_state_dict'])

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    loss_epoch = 0
    for iteration, batch in enumerate(dataloader):
        optimizer.zero_grad()
        rgb,MS = batch[0],batch[1]
        # forward
        rgb = rgb.cuda()
        MS = MS.cuda()


        out = model(rgb)

        l1_loss = L1_loss(out, MS)
        vgg_loss0 = VGG_loss(out[:, 0, :, :].unsqueeze(dim=1), MS[:, 0, :, :].unsqueeze(dim=1))
        vgg_loss1 = VGG_loss(out[:, 1, :, :].unsqueeze(dim=1), MS[:, 1, :, :].unsqueeze(dim=1))
        vgg_loss2 = VGG_loss(out[:, 2, :, :].unsqueeze(dim=1), MS[:, 2, :, :].unsqueeze(dim=1))
        vgg_loss3 = VGG_loss(out[:, 3, :, :].unsqueeze(dim=1), MS[:, 3, :, :].unsqueeze(dim=1))
        vgg_loss4 = VGG_loss(out[:, 4, :, :].unsqueeze(dim=1), MS[:, 4, :, :].unsqueeze(dim=1))
        vgg_loss5 = VGG_loss(out[:, 5, :, :].unsqueeze(dim=1), MS[:, 5, :, :].unsqueeze(dim=1))
        vgg_loss6 = VGG_loss(out[:, 6, :, :].unsqueeze(dim=1), MS[:, 6, :, :].unsqueeze(dim=1))
        vgg_loss7 = VGG_loss(out[:, 7, :, :].unsqueeze(dim=1), MS[:, 7, :, :].unsqueeze(dim=1))
        vgg_loss = (vgg_loss0 + vgg_loss1 + vgg_loss2 + vgg_loss3 + vgg_loss4 + vgg_loss5 + vgg_loss6 + vgg_loss7) / 8.0

        loss = l1_loss + vgg_loss

        # backward
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        print("===> Epoch[{}]({}/{}): vgg_loss: {:.7f} | l1_loss: {:.7f}  | loss: {:.7f}".format(epoch, iteration, len(dataloader), vgg_loss.item(), l1_loss.item(), loss.item()))

    loss_epoch = loss_epoch / (len(dataloader))
    print("===> Epoch[{}]: loss: {:.7f}".format(epoch, loss_epoch))

    record_loss(loss_csv, epoch, loss_epoch)
    scheduler.step(loss)

    # checkpoint
    if epoch % 100 == 0:
        net_path = SAVE_FOLDER + "/net_epoch_{}.pth".format(epoch)
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, net_path)
        print("Checkpoint saved to {}".format(net_path))
