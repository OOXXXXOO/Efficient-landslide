from efnet import *
import numpy as np
import torch

# import torchvision.transforms as transform
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL.Image as Image
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("experiment")


best_acc=0

GPUID=0
os.environ['CUDA_VISIBLE_DEVICES']=str(GPUID)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)




datalist=np.load('/home/winshare/GisProgect/Dataset/dataset2.npy')


print(datalist.shape)

transform_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

])


def default_loader(image):
    # print(image.shape)
    image = np.expand_dims(image, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    # print(image.shape)
    image=image.astype(np.float32)
    # mutil= mutil.astype(np.float32)
    mutil = transform_train(image)
    # print(mutil)
    # exit(0)
    return mutil

class trainset(Dataset):
    def __init__(self,loader=default_loader,train_ratio=0.6):
        self.length=int(len(datalist)*train_ratio)

        self.labelsum=datalist[:self.length,1]
        self.vecsum=datalist[:self.length,2:6]
        self.imagesum=datalist[:self.length,0]
        self.labels=[]
        self.images=[]
        self.vecs=[]
        print('start turn TRAIN data ratio...')
        for index,label in enumerate(tqdm(self.labelsum)):
            if label==0:
                if len(self.labels)!=0 and sum(self.labels)/len(self.labels)>0.49:
                    self.labels.append(self.labelsum[index])
                    self.images.append(self.imagesum[index])
                    self.vecs.append(self.vecsum[index])


            if label==1:
                    self.labels.append(self.labelsum[index])
                    self.images.append(self.imagesum[index])
                    self.vecs.append(self.vecsum[index])
        print('*********\n\n,Positive/All Data=',sum(self.labels),'/',len(self.labels),'\n\n***********')
        self.loader=loader
    def __getitem__(self, index):
        image=self.loader(self.images[index])
        image=image.float()
        label=np.array((self.labels[index]))
        label=torch.from_numpy(label)
        vec=np.array((list(self.vecs[index])))
        vec_=torch.from_numpy(vec)
        vec_=vec_.float()
        return image,vec_,label
    def __len__(self):
        return len(self.labels)


class testset(Dataset):
    def __init__(self,loader=default_loader,test_ratio=0.3):
        self.length=int(len(datalist)*(1-test_ratio))

        self.labelsum=datalist[self.length:,1]
        self.vecsum=datalist[self.length:,2:6]
        self.imagesum=datalist[self.length:,0]
        self.labels=[]
        self.images=[]
        self.vecs=[]
        print('start turn TEST data ratio...')
        for index,label in enumerate(tqdm(self.labelsum)):
            if label==0:
                if len(self.labels)!=0 and sum(self.labels)/len(self.labels)>0.49:
                    self.labels.append(self.labelsum[index])
                    self.images.append(self.imagesum[index])
                    self.vecs.append(self.vecsum[index])


            if label==1:
                    self.labels.append(self.labelsum[index])
                    self.images.append(self.imagesum[index])
                    self.vecs.append(self.vecsum[index])
        print('*********\n\n,Positive/All Data=',sum(self.labels),'/',len(self.labels),'\n\n***********')
        self.loader=loader
    def __getitem__(self, index):
        image=self.loader(self.images[index])
        image=image.float()
        label=np.array((self.labels[index]))
        label=torch.from_numpy(label)
        vec=np.array((list(self.vecs[index])))
        vec_=torch.from_numpy(vec)
        vec_=vec_.float()
        return image,vec_,label
        # return image,vec,label
    def __len__(self):
        return len(self.labels)

traindata=trainset()
trainloader=DataLoader(traindata,batch_size=128,shuffle=True)
testdata=testset()
testloader=DataLoader(testdata,batch_size=128,shuffle=True)





#model

EFnet = EVIBuild()
EFnet = EFnet.to(device)
if device == 'cuda':
    EFnet = torch.nn.DataParallel(EFnet)
    cudnn.benchmark = True
print(EFnet)

#Criterion
# EFnet.load_state_dict(torch.load('./checkpoint/ckpt.pth')['net'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(EFnet.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
trainstep=0
def train(epochs):
    global trainstep
    EFnet.train()
    train_loss=0
    correct=0
    total=0

    for batchid,(input_,vec_,label_) in enumerate(trainloader):
        # print(batchid)
        input_,vec_,label_=input_.to(device),vec_.to(device),label_.to(device)
        images=(input_,vec_)
        optimizer.zero_grad()
        outputs=EFnet(images)
        loss=criterion(outputs,label_)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()

        grid = torchvision.utils.make_grid(input_)
        writer.add_image('train', grid, 0)

        _,predict=outputs.max(1)
        total+=label_.size(0)
        correct+=predict.eq(label_).sum().item()

        progress_bar(batchid, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)| Step:%d'
            % (train_loss/(batchid+1), 100.*correct/total, correct, total,trainstep))

        writer.add_scalars(main_tag='acc', tag_scalar_dict={'train_acc':100. * correct / total}, global_step=trainstep)
        writer.add_scalars(main_tag='loss',tag_scalar_dict= {'train_loss':(train_loss/(batchid+1))},global_step= trainstep)
        trainstep+=1


teststep=0
def test(epoch):
    global best_acc
    global teststep
    EFnet.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batchid,(input_,vec_,label_) in enumerate(testloader):
            # print(batchid)
            input_,vec_,label_=input_.to(device),vec_.to(device),label_.to(device)
            images=(input_,vec_)
            outputs=EFnet(images)

            loss=criterion(outputs,label_)

            test_loss+=loss.item()

            _,predict=outputs.max(1)
            total+=label_.size(0)
            correct+=predict.eq(label_).sum().item()
            grid = torchvision.utils.make_grid(input_)
            writer.add_image('test', grid, 0)


            progress_bar(batchid, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)| Step:%d'
                % (test_loss/(batchid+1), 100.*correct/total, correct, total,teststep))

            writer.add_scalars(main_tag='acc', tag_scalar_dict={'test_acc':100. * correct / total}, global_step=teststep)
            writer.add_scalars(main_tag='loss',tag_scalar_dict= {'test_loss':(test_loss/(batchid+1))},global_step=teststep)
            teststep+=1



    acc = 100. * correct / total
    if acc > best_acc:
        print('\n\nSaving..\n\n')
        state = {
            'net': EFnet.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt2.pth')
        best_acc = acc



for epoch in range(0,500):
    print('||||----epoch : ',epoch,'/500')
    train(epoch)
    test(epoch)
writer.close()