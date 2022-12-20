import csv
import pandas as pd
import cv2
import os

data_dir = '/home/ydm/Downloads/label/'
BASE = '/home/ydm/4-2/project/dataset/scene/dataset/' # 저장할 폴더 
ROOT = '/home/ydm/Downloads/Surface_1' # 데이터셋 폴더 


csv_file = os.listdir(data_dir)
csv_file.sort()
# print(csv_file)
# exit()

for i in range(len(csv_file)):
    if i == 0:
        pass
    else:
        data = pd.read_csv(data_dir + csv_file[i], encoding = 'utf-8') #, sep=";") # index_col = 4)
        # print(data)
        # exit()
        
        # 인도, 차도, 애매 
        sidewalk = BASE + 'sidewalk/'
        road = BASE + 'road/'
        DN = BASE + 'DN/'

        file_name = data['FileName']
        label = data['label']

        count1 = len(os.listdir(sidewalk)) 
        count2 = len(os.listdir(road)) 
        count3 = len(os.listdir(DN)) 
        # print(count1, count2, count3)
        # exit()

        for i in range(len(file_name)):
            name = ROOT + file_name[i]
            image = cv2.imread(name, cv2.IMREAD_UNCHANGED)
            if label[i] == 1:
                count1+=1
                cv2.imwrite(sidewalk + str(count1) + '.jpg', image)
            elif label[i] == 2:
                count2+=1
                cv2.imwrite(road + str(count2) + '.jpg', image)
            else:
                count3+=1
                cv2.imwrite(DN + str(count3) + '.jpg', image)
            