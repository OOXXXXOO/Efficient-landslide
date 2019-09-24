import sys
import os
sys.path.append("./Labelgis")
# import reference
from wsio.iolib import*
import time
import math
from tqdm import tqdm

class TifDataset(TIF):
    def __init__(self,root='./Dataset'):
        super(TifDataset,self).__init__()
        self.root=root
        if not os.path.exists(self.root):
            print('create root in :',self.root)
            os.makedirs(self.root)
    def setdata(self,datapath):
        centre=(103,31.4)
        self.inputdata=self.Read(datapath)
        self.cordcentre=self.lonlat2imagexy(centre[0],centre[1])
        self.max_height=self.inputdata.max()
        print('Max Height is',self.max_height)


    def setlabel(self,labelpath):
        self.labeldata=self.Read(labelpath)
        assert self.labeldata.shape[0]==self.inputdata.shape[0],'Input Shape[0] not equal to Label'
        assert self.labeldata.shape[1]==self.inputdata.shape[1],'Input Shape[1] not equal to Label'
    def setflag(self,flag=str(time.asctime( time.localtime(time.time())))):
        print('\n\n**********\n\nMission Name =',flag,'\n\n**********')
        self.taskname=flag
    def seteqcentre(self,x,y):
        self.eqx=x
        self.eqy=y
    def setintensity(self,intensity):
        self.intensity=self.Read(intensity)

    def random_generate(self,patch_size=32,sample_ratio=0.3,npy=None,csv=None):
        
        self.datapath=os.path.join(self.root,'Data')
        self.labelpath=os.path.join(self.root,'Label')
    
        if not os.path.exists(self.datapath):
            print('create dataset folder in',self.datapath)
            os.makedirs(self.datapath)
        if not os.path.exists(self.labelpath):
            print('create labelset folder in',self.labelpath)
            os.makedirs(self.labelpath)
 
    
        patch=[]
        # with open(self.csv, 'a') as csvfile:
        H=self.inputdata.shape[0]-patch_size
        W=self.inputdata.shape[1]-patch_size
        num=int((H*W)/(patch_size**2*sample_ratio))
        print('Data Length is ,',num)
        if csv!=None:
            self.csv=os.path.join(self.root,csv)
            if not os.path.exists(self.csv):
                print('create index csv in',self.csv)
                os.system('touch '+self.csv)
        if npy!=None:
            self.npy=os.path.join(self.root,npy)
            if not os.path.exists(self.npy):
                print('create index csv in',self.npy)
                os.system('touch '+self.npy)
            print(self.cordcentre)
            for i in tqdm(range(num)):
                x=np.random.randint(0, H)
                y=np.random.randint(0, W)
                image=self.inputdata[x:x+patch_size,y:y+patch_size]/self.max_height
                label=self.labeldata[x:x+patch_size,y:y+patch_size]
                if label.sum()>0:
                    label=1
                else:
                    label=0
                distence=math.sqrt((x-self.cordcentre[0])**2+(y-self.cordcentre[1])**2)/((math.sqrt(2)*(max(H,W))))
                distence=float(distence)
                # print((image,label, self.intensity[x,y]/10,x-self.cordcentre[0]/W,y-self.cordcentre[1]/H,distence))
                patch.append((image,
                             label,
                             self.intensity[x,y]/10,
                             float((x-self.cordcentre[0])/W),
                             float((y-self.cordcentre[1])/H),
                             distence))

        np.save(self.npy,patch)
        print('Save Done in,',self.npy)
        print('DataDemo is :\n',patch[0])













            








def main():
    tifpath='/home/winshare/paper/DEMUPDATE/data/LargeUpdate.tif'
    labelpath='/home/winshare/paper/DEMUPDATE/label/labelupdate.tif'
    intensity='/home/winshare/paper/DEMUPDATE/intensityupdate.tif'
    intensity
    tif=TifDataset()
    tif.setdata(tifpath)
    tif.setlabel(labelpath)
    tif.setflag()
    tif.setintensity(intensity)
    tif.random_generate(npy='dataset2.npy')


if __name__ == '__main__':
    main()
    