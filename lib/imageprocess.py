import cv2

import os

import pandas as pd
import json
traincsv='./train_dem.csv'
testcsv='./test_dem.csv'

train=pd.read_csv(traincsv)
test=pd.read_csv(testcsv)

# # train=train[:]
print(train)
# print(test)


train_label={}
test_label={}

for index,imagedir in enumerate(train.itertuples()):
    print(imagedir[1])
    imagedir='./'+str(imagedir[1])
    image=cv2.imread(imagedir)
    if image.sum()>0:
        print('postive :',imagedir)
        train_label[imagedir[1]] = 'True'
    else:
        print('negative : ',imagedir)
        train_label[imagedir[1]] = 'False'

print(train_label)
with open('./train_label.json',"w") as f:
    train_json=json.dumps(train_label)
    f.write(train_json)
    f.close()
    print('write done')





for imagedir in test.itertuples():
    print(imagedir[1])
    imagedir='./'+str(imagedir[1])
    image=cv2.imread(imagedir)
    if image.sum()>0:
        print('postive :',imagedir)
    else:
        print('negative : ',imagedir)


print(test_label)
with open('./test_label.json',"w") as f:
    train_json=json.dumps(train_label)
    f.write(train_json)
    f.close()
    print('write done')
