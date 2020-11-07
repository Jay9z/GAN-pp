from transform.transforms import *

class TrainAugmentation():
    def __init__(self, image_size, mean_val=0, std_val=1):
        #TODO: add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops
        self.augment = Compose(
                [
                    RandomScale(),
                    Pad(image_size,mean_val=mean_val),
                    RandomCrop(image_size),
                    RandomFlip(),
                    ConvertDataType(),
                    Normalize(mean_val,std_val)
                    ]
                    )


    def __call__(self, image, label=None):
        return self.augment(image,label)

class Normalize2():
    def __init__(self, mean_val=0, std_val=1.0):
        #TODO: add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops
        self.augment = Compose(
                [
                    #ConvertDataType(),
                    Normalize(mean_val,std_val)
                    ]
                    )

    def __call__(self, image, label=None):
        return self.augment(image,label)

class Augmentation1():
    def __init__(self, mean_val=0, std_val=1.0):
        #TODO: add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops
        self.augment = Compose(
                [
                    ConvertDataType(),
                    Normalize(mean_val,std_val)
                    ]
                    )

    def __call__(self, image, label=None):
        return self.augment(image,label)

class Augmentation2():
    def __init__(self, mean_val=0, std_val=1.0):
        #TODO: add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops
        self.augment = Compose(
                [

                    ConvertDataType(),
                    Normalize(mean_val,std_val)
                    ]
                    )

    def __call__(self, image, label=None):
        return self.augment(image,label)

class Augmentation3():
    def __init__(self, mean_val=0, std_val=1.0):
        #TODO: add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops
        self.augment = Compose(
                [
                    RandomFlip(),
                    ConvertDataType(),
                    Normalize(mean_val,std_val)
                    ]
                    )

    def __call__(self, image, label=None):
        return self.augment(image,label)

class Augmentation4():
    def __init__(self, mean_val=0, std_val=1.0):
        #TODO: add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops
        self.augment = Compose(
                [
                    RandomScale(),
                    ConvertDataType(),
                    Normalize(mean_val,std_val)
                    ]
                    )

    def __call__(self, image, label=None):
        return self.augment(image,label)


def reconstruct(x,GE_x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    gex_mean = np.mean(GE_x)
    gex_std = np.std(GE_x)
    xx = x_mean + (GE_x-gex_mean)*gex_std/x_std
    xx = (xx - np.min(xx))/(np.max(xx)-np.min(xx))*255
    return xx