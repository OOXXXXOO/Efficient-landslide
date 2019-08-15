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
import tifffile as tif
import matplotlib.pyplot as plt
transform_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

])
class predict():
    def __init__(self):
        self.cord=(3600,5760)
        return
        
    def setintensity(self,intensity):
        self.intensity=tif.imread(intensity)
        # plt.imshow(self.intensity),plt.show()

    def netload(self,model_path='./checkpoint/ckpt2.pth'):
        GPUID=0
        os.environ['CUDA_VISIBLE_DEVICES']=str(GPUID)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        EFnet = EVIBuild()
        EFnet = EFnet.to(self.device)
        EFnet.eval()

        if self.device == 'cuda':
            EFnet = torch.nn.DataParallel(EFnet)
            cudnn.benchmark = True
        EFnet.load_state_dict(torch.load(model_path)['net'])
        self.net=EFnet
        # self.net.eval()
        self.loaded=True

    def default_loader(self,image):
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
    def despimage(self,tifpath,_size=32,rate=0.9,random=True):
        image=tif.imread(tifpath)
        self.maxheight=image.max()
        # plt.imshow(image),plt.show()
        W=image.shape[0]-_size
        H=image.shape[1]-_size
        print(image.shape)
        result=np.zeros(image.shape,dtype=np.float32)
        num=int(W*H/(_size**2*rate))
        if not random:
            for cx in tqdm(range(0,W,16)):
                for cy in range(0,H,16):
                    small=image[cx:cx+_size,cy:cy+_size]
                    small=small/self.maxheight
            

                    input_=self.default_loader(small)
                    intensity=self.intensity[cx,cy]
                    xx=(cx-self.cord[0])
                    yy=(cy-self.cord[1])
                    distence=math.sqrt((xx)**2+(yy)**2)/(math.sqrt(2)*H)
                    vec=(intensity/10,xx/W,yy/W,distence)
                    # if i==0:
                    #     print(input_,' || ',vec)
                    vec=torch.Tensor(vec)
                    vec=vec.float()
                    
                    # print(input_.shape,' || ',vec.shape)
                    input_=input_.unsqueeze(0)
                    vec=vec.unsqueeze(0)
                    # print(input_.shape,' || ',vec.shape)

                    input_,vec=input_.to(self.device),vec.to(self.device)
                    input_assembled=(input_,vec)
                    # print('cord:,',x,y,image[x,y],self.maxheight,input_)
                    
                    if self.loaded:
                        outputs=self.net(input_assembled)
                        _,predict=outputs.max(1)
                        # print(predict.cpu().numpy()[0])
                        # exit(0)

                        result[cx:cx+_size,cy:cy+_size]+=predict.cpu().numpy()[0]



        if random:
            for i in tqdm(range(num)):
                x=np.random.randint(0,W)
                y=np.random.randint(0,H)
                small=image[x:x+_size,y:y+_size]
                small=small/self.maxheight
                # print(small)
                # exit(0)
                input_=self.default_loader(small)
                intensity=self.intensity[x,y]
                xx=(x-self.cord[0])
                yy=(y-self.cord[1])
                distence=math.sqrt((xx)**2+(yy)**2)/(math.sqrt(2)*H)
                vec=(intensity/10,xx/W,yy/W,distence)
                if i==0:
                    print(input_,' || ',vec)
                vec=torch.Tensor(vec)
                vec=vec.float()
                
                # print(input_.shape,' || ',vec.shape)
                input_=input_.unsqueeze(0)
                vec=vec.unsqueeze(0)
                # print(input_.shape,' || ',vec.shape)

                input_,vec=input_.to(self.device),vec.to(self.device)
                input_assembled=(input_,vec)
                # print('cord:,',x,y,image[x,y],self.maxheight,input_)
                
                if self.loaded:
                    outputs=self.net(input_assembled)
                    _,predict=outputs.max(1)
                    # print(predict)
                    result[x:x+_size,y:y+_size]+=predict.cpu().numpy()[0]
                

            
         
            # image[x:x+_size,y:y+_size]+=np.ones((_size,_size),dtype=float)

        plt.imshow(result),plt.show()

            # patch=image


def main():
    dempath='/home/winshare/paper/DEMUPDATE/data/LargeUpdate.tif'
    intensity='/home/winshare/paper/DEMUPDATE/intensityupdate.tif'
    predict_=predict()
    predict_.netload(model_path='/home/winshare/paper/DemClassfier/checkpoint/ckpt2.pth')
    predict_.setintensity(intensity)
    predict_.despimage(dempath)
    image=torch.randn((1,3,32,32))
    vec=torch.randn((1,4))
    print(image,'\n\n',vec)
    a=(image.to(predict_.device),vec.to(predict_.device))
    b=predict_.net(a)
    print(b)



if __name__ == '__main__':
    main()    
    

    



