import cv2
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, label=None):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class Normalize(object):
    def __init__(self, mean_val, std_val, val_scale=1):
        # set val_scale = 1 if mean and std are in range (0,1)
        # set val_scale to other value, if mean and std are in range (0,255)
        self.mean = np.array(mean_val, dtype=np.float32)
        self.std = np.array(std_val, dtype=np.float32)
        self.val_scale = 1/255.0 if val_scale==1 else 1
    def __call__(self, image, label=None):
        image = image.astype(np.float32)
        image = image * self.val_scale
        image = image - self.mean
        image = image * (1 / self.std)
        return image, label


class ConvertDataType(object):
    def __call__(self, image, label=None):
        if label is not None:
            label = label.astype(np.int64)
        return image.astype(np.float32), label


class Pad(object):
    def __init__(self, size, ignore_label=255, mean_val=0, val_scale=1):
        # set val_scale to 1 if mean_val is in range (0, 1)
        # set val_scale to 255 if mean_val is in range (0, 255) 
        factor = 255 if val_scale == 1 else 1

        self.size = size
        self.ignore_label = ignore_label
        self.mean_val=mean_val
        # from 0-1 to 0-255
        if isinstance(self.mean_val, (tuple,list)):
            self.mean_val = [int(x* factor) for x in self.mean_val]
        else:
            self.mean_val = int(self.mean_val * factor)


    def __call__(self, image, label=None):
        h, w, c = image.shape
        pad_h = max(self.size - h, 0)
        pad_w = max(self.size - w, 0)

        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if pad_h > 0 or pad_w > 0:

            image = cv2.copyMakeBorder(image,
                                       top=pad_h_half,
                                       left=pad_w_half,
                                       bottom=pad_h - pad_h_half,
                                       right=pad_w - pad_w_half,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=self.mean_val)
            if label is not None:
                label = cv2.copyMakeBorder(label,
                                           top=pad_h_half,
                                           left=pad_w_half,
                                           bottom=pad_h - pad_h_half,
                                           right=pad_w - pad_w_half,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=self.ignore_label)
        return image, label


# TODO
class CenterCrop(object):
    def __init__(self,size):
        self.size = size

    def __call__(self,image,label=None):
        h,w,c = image.shape
        left_r,left_c = (h-self.size)//2,(w-self.size)//2
        img = image[left_r:left_r+self.size,left_c:left_c+self.size,:]
        lab = label[left_r:left_r+self.size,left_c:left_c+self.size]
        return img,lab

# TODO
class Resize(object):
    def __init__(self,size):
        self.size = size

    def __call__(self,image,label=None):
        img = cv2.resize(image,(self.size,self.size),cv2.INTER_LINEAR)
        if label is not None:
            lab = cv2.resize(label,(self.size,self.size),cv2.INTER_NEAREST)
        return img,lab


# TODO
class RandomFlip(object):
    def __call__(self,image,label=None):
        val= np.random.rand()*2
        if val<1:
            image = np.flip(image,axis=0)
            if label is not None:
                label = np.flip(label,axis=0)
        elif val<2:
            image = np.flip(image,axis=1)
            if label is not None:
                label = np.flip(label,axis=1)

        return image,label

# TODO
class RandomCrop(object):
    def __init__(self,size):
        self.size = size
    
    def __call__(self,image,label=None):
        h,w,c = image.shape
        scales = np.random.rand(2)
        left_r,left_c = int(scales[0]*(h-self.size)-0.5), int(scales[1]*(w-self.size)-0.5)
        img = image[left_r:left_r+self.size, left_c:left_c+self.size,:]
        if label is not None:
            label = label[left_r:left_r+self.size, left_c:left_c+self.size]
        return img,label


# TODO
class Scale(object):
    def __init__(self,scale=1.0):
        self.scale = scale

    def __call__(self,image,label=None):
        h_1,w_1,c_1 = image.shape
        h_2,w_2 = label.shape
        assert h_1 == h_2, "wrong height"
        assert w_1 == w_2, "wrong width"
        h,w = int(h_1*self.scale),int(w_1*self.scale)
        image = cv2.resize(image,(h,w),interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label,(h,w),interpolation=cv2.INTER_NEAREST)
        return image,label


# TODO
class RandomScale(object):
    def __init__(self,min_scale=0.8,max_scale=1.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        assert self.max_scale>self.min_scale,"max_scale must larger than min_scale"
        self.scale_obj = Scale()

    def __call__(self,image,label=None):
        val = np.random.rand()*(self.max_scale-self.min_scale)+self.min_scale
        self.scale_obj.scale = val
        return self.scale_obj(image,label)


def main():
    #image = cv2.imread('./dummy_data/JPEGImages/2008_000064.jpg')
    #label = cv2.imread('./dummy_data/GroundTruth_trainval_png/2008_000064.png')

    image = cv2.imread('work/dummy_data/JPEGImages/2008_000064.jpg',cv2.IMREAD_COLOR)
    label = cv2.imread('work/dummy_data/GroundTruth_trainval_png/2008_000064.png',cv2.IMREAD_GRAYSCALE)
    print(image.shape,label.shape)
    # TODO: crop_size
    crop_size = 256
    # TODO: Transform: RandomSacle, RandomFlip, Pad, RandomCrop
    transf = Compose([
        RandomScale(),
        RandomFlip(),
        Pad(crop_size),
        RandomCrop(crop_size)
    ])

    for i in range(10):
        # TODO: call transform
        img, lab = transf(image,label)
        # TODO: save image
        cv2.imwrite(f"image_{i}.jpg",img)
        cv2.imwrite(f"label_{i}.png",lab)

if __name__ == "__main__":
    main()
