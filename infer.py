
from PIL import Image
import cv2
import os
import numpy as np
import argparse

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.optimizer import AdamOptimizer

from dataloader import DataLoader, Transform
from preprocessing import *
from utils.utils import AverageMeter
from net.idcgan import Generator,Invertor

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='basic')
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


# this inference code reads a list of image path, and do prediction for each image one by one
def generate_images():

    with fluid.dygraph.guard():
        G = Generator()
        pretrain_file = os.path.join(args.checkpoint_folder, f"G_{args.net}-Epoch-{args.num_epochs}")
        print(pretrain_file)
        if os.path.exists(pretrain_file):
            state,_ = fluid.load_dygraph(pretrain_file)
            model.set_dict(state)

        image_folder = ""
        image_list_file = "dummy_data/fabric_list.txt"

        transform = Transform()

        data = DataLoader(image_folder,image_list_file,transform=transform)
        dataloader = fluid.io.DataLoader.from_generator(capacity=2, return_list=True)
        dataloader.set_sample_generator(data,1)

        G.eval()
        for i in range(20):
            z = np.random.rand(1,64)
            z = to_variable(z)
            z = fluid.layers.cast(z,dtype='float32')
            fake_x = G(z)
            result = np.squeeze(fake_x.numpy()[0])
            result_min= np.min(result)
            result_range = np.max(result)-np.min(result)
            result = (result-result_min)/result_range*255
            o_file = f"{args.checkpoint_folder}/gen_{i}.png"
            cv2.imwrite(o_file,result)

def main():

    with fluid.dygraph.guard():
        G = Generator()
        E = Invertor()

        G_pretrain_file = os.path.join(args.checkpoint_folder, f"G_{args.net}-Epoch-{args.num_epochs}")
        E_pretrain_file = os.path.join(args.checkpoint_folder, f"E_{args.net}-Epoch-{args.num_epochs}")
        print(G_pretrain_file)
        print(E_pretrain_file)
        if os.path.exists(G_pretrain_file):
            state,_ = fluid.load_dygraph(G_pretrain_file)
            G.set_dict(state)
        if os.path.exists(E_pretrain_file):
            state,_ = fluid.load_dygraph(E_pretrain_file)
            E.set_dict(state)

        image_folder = ""
        image_list_file = "dummy_data/fabric_list.txt"

        transform = Transform()

        data = DataLoader(image_folder,image_list_file,transform=transform)
        dataloader = fluid.io.DataLoader.from_generator(capacity=2, return_list=True)
        dataloader.set_sample_generator(data,1)

        E.eval()
        G.eval()
        for idx,(images,labels) in enumerate(dataloader):

            x = to_variable(images)
            x = fluid.layers.cast(x,dtype='float32')
            x = fluid.layers.transpose(x,perm=[0,3,1,2])

            reconstruct_x = G(E(x))
            reconstruct_image = fluid.layers.transpose(reconstruct_x,perm=[0,2,3,1])
            result = np.squeeze(reconstruct_image.numpy()[0])*255

            o_file = f"{args.checkpoint_folder}/x_{idx}.png"
            result = reconstruct(np.squeeze(images.numpy()[0]),result)
            cv2.imwrite(o_file,result)
            if idx == 20:
                break

if __name__ == "__main__":
    generate_images()
    main()
