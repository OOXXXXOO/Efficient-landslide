from efnet import *
# import torchvision.transforms as transform
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL.Image as Image
import torch.backends.cudnn as cudnn
import torch.optim as optim

GPUID=1
os.environ['CUDA_VISIBLE_DEVICES']=str(GPUID)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# net19 = vgg19_bn()
# print(net19)


dataroot='/workspace/UNetXS/'
demcsv='./train_dem.csv'
labeljson='./train_label.json'
csv= pd.read_csv(demcsv.format('_dem'), header=None)


with open(labeljson,'r') as jsonfile:
    label=json.load(jsonfile)
    # print(label)
    jsonfile.close()

imagelist=[]
labellist=[]
for index,dem in enumerate(csv.itertuples()):
    if index <6449 :
        # print('process with :',dem[1])
        # print('label is ',label[str(index)])
        imagelist.append(os.path.join(dataroot,dem[1]))
        if label[str(index)]=='True':
            labellist.append(1)
        else:
            labellist.append(0)
best_acc=0


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
def default_loader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.convert('RGB')
    # img_pil = img_pil.resize((32,32))
    img_tensor = transform_train(img_pil)
    return img_tensor

class trainset(Dataset):
    def __init__(self,loader=default_loader):
        self.images=imagelist[:6000]
        self.labels=labellist[:6000]
        self.loader=loader
    def __getitem__(self, index):

        fn=self.images[index]
        img=self.loader(fn)
        labels=self.labels[index]
        labels=np.array(labels)
        labels=torch.from_numpy(labels)
        return img,labels

    def __len__(self):
        return len(self.images)

class testset(Dataset):
    def __init__(self,loader=default_loader):
        self.images=imagelist[6000:]
        self.labels=labellist[6000:]
        self.loader=loader



    def __getitem__(self, index):

        fn=self.images[index]
        img=self.loader(fn)
        labels=self.labels[index]
        labels = np.array(labels)
        labels = torch.from_numpy(labels)
        return img,labels

    def __len__(self):
        return len(self.images)



traindata=trainset()
trainloader=DataLoader(traindata,batch_size=32,shuffle=True)

testdata=testset()
testloader=DataLoader(testdata,batch_size=32,shuffle=True)



#model

EFnet = EfficientNetB0()
EFnet = EFnet.to(device)
if device == 'cuda':
    EFnet = torch.nn.DataParallel(EFnet)
    cudnn.benchmark = True

#Criterion

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(EFnet.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

def train(epochs):
    EFnet.train()
    train_loss=0
    correct=0
    total=0

    for batchid,(inputs_,labels_) in enumerate(trainloader):
        print(batchid)
        # print(inputs_.shape)
        # print(labels_.shape)
        inputs_,labels_=inputs_.to(device),labels_.to(device)
        optimizer.zero_grad()
        outputs=EFnet(inputs_)
        # print(outputs,'\n',labels_,'\n\n')
        loss=criterion(outputs,labels_)
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        # print('output:',outputs)
        # print('output shape:',outputs.shape)

        _,predict=outputs.max(1)
        total+=labels_.size(0)
        correct+=predict.eq(labels_).sum().item()

        progress_bar(batchid, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batchid+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    EFnet.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = EFnet(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

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
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc



for epoch in range(0,500):
    print('||||----epoch : ',epoch,'/200')
    train(epoch)
    test(epoch)