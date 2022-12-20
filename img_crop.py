import csv
from PIL import Image
import os

data_dir = '/home/ydm/4-2/project/dataset/train/general/'
BASE = '/home/ydm/4-2/project/dataset/scene/origin_data/' # 원본 이미지 저장 
ROOT = '/home/ydm/4-2/project/dataset/scene/crop_data/' # Croped 이미지 저장 

# PARAM
limit = 70
left = 325
up = 408
rigth = 1595
down = 1208

csv_file = os.listdir(data_dir)
# csv_file.sort()
dataset = [] 

for i in range(len(csv_file)):
    path = data_dir + csv_file[i]
    file = os.listdir(path)
    count = 0
    for j in range(len(file)):
        count+=1
        if '3' in csv_file[i]:
            break
        elif count > limit:
            break
        else:
            file_name = os.path.join(path, file[j])
            # print(file_name)
            dataset.append(file_name)
            # print(file_name)
# print(len(dataset))
num = 0
print("We save the image!")
print("Wait a moment..")
for i in range(len(dataset)):
    image1 = Image.open(dataset[i])
    # print(image1.size)
    num+=1
    croppedImage = image1.crop((left,up, rigth, down))
    image1.save(BASE + str(num) + '.png', 'png')
    croppedImage.save(ROOT + str(num) + '.png', 'png')
    # print(croppedImage.size)
    # image1.show()
    # croppedImage.show()
print("Finish!")
# #이미지의 크기 출력
# print(image1.size)