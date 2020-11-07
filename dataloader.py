import os
import random
import numpy as np
import cv2
import paddle.fluid as fluid
from PIL import Image
class Transform():
    def __init__(self,size=32):
        self.size=size
    
    def __call__(self,image,label):
        img = image/255.0
        return img,label


class DataLoader():
    def __init__(self,
                 image_folder,
                 image_list_file,
                 transform=None,
                 shuffle=True):
        self.image_folder = image_folder
        self.image_list_file = image_list_file
        self.transform = transform
        self.shuffle = shuffle
        self.paths = self.read_list()

    def read_list(self):
        fp = open(self.image_list_file)
        path_list = []
        for line in fp.readlines():
            paths = line.split()
            image_path = paths[0]
            image_path = os.path.join(self.image_folder,image_path)
            path_list.append(image_path)
        if self.shuffle:
            random.shuffle(path_list)
        return path_list

    def preprocess(self, data,label):
        if self.transform:
            data,label = self.transform(data,label)
        #print(data.shape)
        data = data[:,:,np.newaxis]
        return data,label      ## must be a list or tuple to compatible with function,from_generator()

    def __len__(self):
        return len(self.paths)

    def __call__(self):
        for image_path in self.paths:
            data = np.array(Image.open(image_path))
            label = np.ones(1)
            yield self.preprocess(data,label)


def make_train_dataset(image_path,folder):
    img = Image.open(image_path)
    img = np.array(img)
    img = cv2.blur(img,(3,3))
    print(img.shape)
    fp = open("fabric_list.txt",'a+')
    for i in range(2,16):
        for j in range(500):
            x,y = np.random.rand(2)*(256-32)
            x,y = int(x),int(y)
            temp = img[y:y+32,i*256+x:i*256+x+32]
            temp2 = Image.fromarray(np.uint8(temp))
            image_path = f"{folder}/{i}_{j}.png"
            temp2.save(image_path)
            fp.write(image_path)
            fp.write("\n")
            fp.flush()

def main():
    batch_size = 3
    #place = fluid.CPUPlace()
    transform = Transform()
    #place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard():
        # TODO: craete BasicDataloder instance
        image_folder=""
        image_list_file="dummy_data/fabric_list.txt"
        data = DataLoader(image_folder,image_list_file,transform=transform)
        #z = ZLoader(len(data))
        #print("size of data: ",len(z))

        dataloader = fluid.io.DataLoader.from_generator(capacity=2, return_list=True)
        dataloader.set_sample_generator(data,batch_size)

        num_epoch = 2
        for epoch in range(1, num_epoch+1):
            print(f'Epoch [{epoch}/{num_epoch}]:')
            for idx, (data,label) in enumerate(dataloader):
                print(f'Iter {idx}, Data shape: {data.shape},{label.shape} ')
                img = np.squeeze(data.numpy()[0])*255
                cv2.imwrite(f"{idx}.png",img)

if __name__ == "__main__":
    
    main()

    # image_path = "dummy_data/fabric/0001_000_02.png"
    # save_folder = "dummy_data/fabric/train"
    # make_train_dataset(image_path,save_folder)

    # image_path = "dummy_data/fabric/0011_006_02.png"
    # save_folder = "dummy_data/fabric/test"
    # make_dataset(image_path,save_folder)