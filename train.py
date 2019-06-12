from vggclassfier import *
# import torchvision.transforms as transform
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL.Image as Image
GPUID=1
os.environ['CUDA_VISIBLE_DEVICES']=str(GPUID)
net19 = vgg19_bn()
print(net19)


dataroot='/workspace/UNetXS/'
demcsv='/workspace/UNetXS/train_dem.csv'
labeljson='/workspace/UNetXS/train_label.json'
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



normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
def default_loader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.resize((224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor

class trainset(Dataset):
    def __init__(self,loader=default_loader):
        self.images=imagelist[:6000]
        self.target=labellist[:6000]
        self.loader=loader
    def __getitem__(self, index):
        if index < 6000:
            fn=self.images[index]
            img=self.loader(fn)
            target=self.target[index]
            return img,target
        else:
            return self.loader(self.images[0]),self.target[0]
    def __len__(self):
        return len(self.images)



traindata=trainset()
trainloader=DataLoader(traindata,batch_size=4,shuffle=True)


for data in trainloader:
    print('train data is ',data)


