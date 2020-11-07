
from PIL import Image
import cv2
import os
import numpy as np
import argparse

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.optimizer import AdamOptimizer

from dataloader import DataLoader
from preprocessing import *
from utils.utils import AverageMeter
from net.idcgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='G_basic')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_folder', type=str, default='./dummy_data')
parser.add_argument('--image_list_file', type=str, default='./dummy_data/list.txt')
parser.add_argument('--checkpoint_folder', type=str, default='./output')
parser.add_argument('--save_freq', type=int, default=2)
parser.add_argument('--inf_type', type=int, default=0)

args = parser.parse_args()

def colorize(gray, palette=2):
    '''
    COLORMAP_AUTUMN = 0,
    COLORMAP_BONE = 1,
    COLORMAP_JET = 2,
    COLORMAP_WINTER = 3,
    COLORMAP_RAINBOW = 4,
    COLORMAP_OCEAN = 5,
    COLORMAP_SUMMER = 6,
    COLORMAP_SPRING = 7,
    COLORMAP_COOL = 8,
    COLORMAP_HSV = 9,
    COLORMAP_PINK = 10,
    COLORMAP_HOT = 11
    '''
    # gray: numpy array of the label and 1*3N size list palette
    # color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    # color.save("color_input.png")
    # data = np.array(color)
    # print(np.min(data),np.max(data),data.shape)
    # #color.putpalette(palette)
    gray = gray.astype(np.uint8)
    color = cv2.applyColorMap(gray,2)
    return color

def save_blend_image(image_file, pred_file):
    image1 = Image.open(image_file)
    image2 = Image.open(pred_file)
    image1 = image1.convert('RGBA')
    image2 = image2.convert('RGBA')
    image = Image.blend(image1, image2, 0.5)
    o_file = pred_file[0:-4] + "_blend.png"
    image.save(o_file)


def inference_resize(model,images):
    tm = Compose([
        Resize(224)  
        ]
        )

    return tm(images)[0]

def inference_sliding():
    pass

def inference_multi_scale():
    pass


# def save_images(images,suffix="input"):
#     ## images
#     # print(images.shape,type(images))
#     if isinstance(images,Image.Image):
#         images = np.array(images)
#     assert isinstance(images,np.ndarray), "wrong image data type" 
#     print(images.shape)
#     n = images.shape[0]
#     for i in range(n):
#         #image = np.transpose(images[i],(1,2,0))
#         image = images[i]
#         image = Image.fromarray(image.astype(np.uint8))#.convert('P')
#         image.save(f"{i}_{suffix}.png")


# this inference code reads a list of image path, and do prediction for each image one by one
def main():
    # 0. env preparation
    with fluid.dygraph.guard():
        # 1. create model
        model = Generator()
        model.eval()
        # 2. load pretrained model 
        pretrain_file = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{args.num_epochs}")
        print(pretrain_file)
        if os.path.exists(pretrain_file):
            state,_ = fluid.load_dygraph(pretrain_file)
            model.set_dict(state)

        # 3. read test image list
        image_folder = ""
        image_list_file = "dummy_data/fabric_list.txt"

        # 4. create transforms for test image, transform should be same as training
        transform = None
        if args.inf_type == 0:
            #transform = Normalize2()
            pass

        data = DataLoader(image_folder,image_list_file,transform=transform)
        dataloader = fluid.io.DataLoader.from_generator(capacity=2, return_list=True)
        dataloader.set_sample_generator(data,1)

        # 5. loop over list of images
        for i in range(20):
        #for images,labels in dataloader():
            z = np.random.rand(1,64)

            # 7. image to variable
            z = to_variable(z)
            z = fluid.layers.cast(z,dtype='float32')

            # 8. call inference func
            preds = model(z).numpy()
            result = np.squeeze(preds[0])
            print(result)

            #print(preds.shape,result.shape)
            o_file = f"{args.checkpoint_folder}/{i}_pred.png"
            cv2.imwrite(o_file,result)

if __name__ == "__main__":
    main()
