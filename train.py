import argparse
import torch.backends.cudnn as cudnn
from dataloader import *
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import socket
import time
from loss import *
from torchvision.transforms import *
from model.model import *

# Training settings
parser = argparse.ArgumentParser(description='MS_Color_constancy')
parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate for adam')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--gpu_mode', default=True, help='use cuda?')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--checktest', default='checktest/MS_CC_Net/', type=str, help='Location to save checkpoint models')
parser.add_argument('--save_folder', default='weights/MS_CC_Net/', type=str, help='Location to save checkpoint models')
parser.add_argument('--data_dir', default='dataset/nus_dataset/', type=str, help='Dataset dir')
parser.add_argument('--fold_num', default=0, type=int, help='Fold_number')
parser.add_argument('--resume', type=bool, default=False, help='patch size')
parser.add_argument('--start_iter', type=int, default=1, help='start_iter')
parser.add_argument('--nEpochs', type=int, default=1000, help='patch size')

opt = parser.parse_args()

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

hostname = str(socket.gethostname())
cudnn.benchmark = True
device = "cuda:0"

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)
if not os.path.exists(opt.checktest):
    os.makedirs(opt.checktest)

def log(text, LOGGER_FILE):
    with open(LOGGER_FILE, 'a') as f:
        f.write(text)
        f.close()

def train(epoch):

    epoch_loss = 0
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        RGB_input,rgb_area,rgb_patch,MS_input,MS_area,MS_patch,gt = Variable(batch[0]), Variable(batch[1]),Variable(batch[2]),Variable(batch[3]),Variable(batch[4]),Variable(batch[5]),Variable(batch[6])#,Variable(batch[7])

        if cuda:
            RGB_input = RGB_input.to(device)
            MS_input = MS_input.to(device)
            rgb_area = rgb_area.to(device)
            MS_area = MS_area.to(device)
            rgb_patch = rgb_patch.to(device)
            MS_patch = MS_patch.to(device)
            gt = gt.to(device)
        optimizer.zero_grad()
        t0 = time.time()

        full_pred, full_rgb, full_confidence, full_illuminant = model(RGB_input, MS_input)
        area_pred, area_rgb, area_confidence, area_illuminant = model(rgb_area, MS_area)
        patch_pred, patch_rgb, patch_confidence, patch_illuminant = model(rgb_patch, MS_patch)

        full_loss = angular_loss(full_pred, gt)
        area_loss = angular_loss(area_pred, gt)
        patch_loss = angular_loss(patch_pred, gt)

        full_area_loss = L1_loss(full_pred, area_pred)
        full_patch_loss = L1_loss(full_pred, patch_pred)
        patch_area_loss = L1_loss(patch_pred, area_pred)
        invariant_loss = full_area_loss + full_patch_loss + patch_area_loss

        contrastive_loss = simclr_loss(full_illuminant, area_illuminant, patch_illuminant) #+ simclr_loss(full_illuminant, patch_illuminant) + simclr_loss(area_illuminant, patch_illuminant)
        loss = full_loss + area_loss + patch_loss + 0.5 * invariant_loss+ 0.5 * contrastive_loss

        epoch_loss += loss.mean().item()
        loss.mean().backward()
        optimizer.step()
        t1 = time.time()
        print(f'===> Epoch[{epoch}]({iteration}/{len(training_data_loader)}):  total loss: {loss.mean()} cc loss:{full_loss}|| Timer: {t1-t0}')
    print(f'===> Epoch {epoch} Complete: Avg. Loss: {epoch_loss / len(training_data_loader)}')

def eval(epoch):
    angular_list=[]
    avg_cc = 0
    model.eval()
    for i,batch in enumerate(testing_data_loader):
        with torch.no_grad():
            RGB_input,MS_input,gt = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

        if cuda:
            RGB_input = RGB_input.to(device)
            MS_input = MS_input.to(device)
            gt = gt.to(device)

        with torch.no_grad():
            full_pred, full_rgb, full_confidence, full_illuminant = model(RGB_input,MS_input)

        ''' calculate AE '''
        cc = angular_loss(full_pred, gt)

        avg_cc += cc
        angular_list.append(cc)
        print('{}번째 angular error = {}'.format(i, cc))
    angular_list1 = angular_list.copy()
    index_min = []
    index_max = []
    '''AE measure'''
    angular_list.sort()
    onefourth = len(testing_data_loader) // 4
    median = angular_list[len(testing_data_loader) // 2]
    maxerror, minerror = 0, 0
    # min 25% error
    for i in range(onefourth):
        minerror += angular_list[i]
        index_min.append(angular_list1.index(angular_list[i]))
    # max 25% error
    angular_list.sort(reverse=True)
    for i in range(onefourth):
        maxerror += angular_list[i]
        index_max.append(angular_list1.index(angular_list[i]))

    maxerror = maxerror / onefourth
    minerror = minerror / onefourth
    avg_cc = avg_cc / len(testing_data_loader)
    print('average:{} | median:{} | worst:{} | Best:{}'.format(avg_cc,median,maxerror,minerror))
    print("===> Processing Done")
    log('Epoch[{}] : average = {}| median:{} | worst:{} | Best:{}\n'.format(epoch,avg_cc,median,maxerror,minerror) , logfile)

def checkpoint(epoch):
    model_out_path = opt.save_folder + "/epoch_{}.pth".format(epoch)
    optimizer_path = opt.save_folder + "/optimizer_epoch_{}.pth".format(epoch)

    torch.save(optimizer.state_dict(),optimizer_path)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def transform():
    return Compose([
        ToTensor(),
    ])

if __name__ == '__main__':
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    ''' Dataset '''
    print('===> Loading datasets')
    training_set = DatasetFromFolder_NUS_fold(data_dir=opt.data_dir, isTrain=True,fold_num=opt.fold_num,transform=transform())
    training_data_loader = DataLoader(training_set, batch_size=opt.train_batch_size, shuffle=True, num_workers=1, drop_last=True)
    testing_set = DatasetFromFolder_NUS_fold(data_dir=opt.data_dir, isTrain=False,fold_num=opt.fold_num,transform=transform())
    testing_data_loader = DataLoader(testing_set,batch_size=opt.test_batch_size,shuffle=False,num_workers=1)

    ''' Model '''
    model = MS_CC_Net(RGB_ch=3, MS_ch=8)
    model = torch.nn.DataParallel(model, device_ids=[0])

    ''' Loss '''
    CC_loss = Angular_loss()
    L1_loss = nn.L1Loss()
    # simclr_loss = SimCLR_Loss(batch_size=opt.train_batch_size,temperature=0.5)
    simclr_loss = SimCLR_3_Loss(batch_size=opt.train_batch_size,temperature=0.5)
    if cuda:
        model = model.to(device)
        L1_loss = L1_loss.to(device)
        simclr_loss = simclr_loss.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

    def lambda_(epoch):
        return pow((1 - ((epoch) / opt.nEpochs)), 0.9)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_,verbose = True)

    logfile = opt.checktest + 'eval.txt'

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters:{total_params}")
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        print('Learning rate : {}'.format([i['lr'] for i in optimizer.param_groups]))

        train(epoch)
        if epoch > 100:
            scheduler.step()
        if (epoch + 1) % 10 == 0:
            eval(epoch)
        if epoch % 100 == 0:
            checkpoint(epoch)

