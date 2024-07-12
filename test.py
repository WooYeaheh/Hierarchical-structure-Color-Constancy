import argparse
from dataloader import *
from torch.utils.data import DataLoader
from loss import *
from torchvision.transforms import *
from model.model import *
import multiprocessing


parser = argparse.ArgumentParser(description='MS_Color_constancy')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_dir', default='dataset/nus_dataset/', type=str, help='Dataset dir')
parser.add_argument('--model_path', default='weights/NUS_proposed/model.pth', type=str, help='model weight dir')
parser.add_argument('--fold_num', default=0, type=int, help='Fold_number')

opt = parser.parse_args()

# device
device = torch.device("cuda:0" if opt.cuda else "cpu")
print(device)

# dataset
def transform_test():
    return Compose([
        ToTensor()
    ])

test_set = DatasetFromFolder_NUS_fold(data_dir=opt.data_dir, transform=transform_test(), isTrain=False)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                 shuffle=False, drop_last=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model = MS_CC_Net(RGB_ch=3, MS_ch=8)
    model = torch.nn.DataParallel(model, device_ids=[0])
    # load weight
    if os.path.isfile(opt.model_path):
        checkpoint = torch.load(opt.model_path)
        model.load_state_dict(checkpoint) # ['model_state_dict']
        print('successfully loaded checkpoints!')
    else:
        print("check model path")
        assert os.path.isfile(opt.model_path) == True

    print('successfully loaded checkpoints!')

    excel_i=[]
    excel_cc=[]
    angular_list = []
    with torch.no_grad():
        avg_cc = 0
        for i,batch in enumerate(testing_data_loader):
            with torch.no_grad():
                RGB_input, MS_input, gt = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

            RGB = RGB_input.to(device)
            MS_input = MS_input.to(device)
            gt = gt.to(device)

            with torch.no_grad():
                pred, est_ilu, confidence_map,_ = model(RGB,MS_input)
            B,C,H,W=RGB.size()
            pred = pred.unsqueeze(2).unsqueeze(2)

            ''' calculate AE '''
            cc = angular_loss(pred, gt)
            avg_cc += cc
            angular_list.append(cc)
            print('Image {} angular error = {}'.format(i,cc))
            excel_i.append(i+1)
            excel_cc.append(float(np.array(cc.detach().cpu())))
        angular_list1 = angular_list.copy()
        index_min = []
        index_max = []
        '''AE measure'''
        angular_list.sort()
        onefourth = len(testing_data_loader)//4
        median = angular_list[len(testing_data_loader)//2]
        maxerror,minerror = 0,0
        # min 25% error
        for i in range(onefourth):
            minerror += angular_list[i]
            index_min.append(angular_list1.index(angular_list[i]))
        # max 25% error
        angular_list.sort(reverse=True)
        for i in range(onefourth):
            maxerror += angular_list[i]
            index_max.append(angular_list1.index(angular_list[i]))

        maxerror = maxerror/onefourth
        minerror = minerror/onefourth
        avg_cc = avg_cc / len(testing_data_loader)
        print('median angular error = {}'.format(median))
        print('average angular error = {}'.format(avg_cc))
        print('max 25% angular error = {}'.format(maxerror))
        print('min 25% angular error = {}'.format(minerror))





