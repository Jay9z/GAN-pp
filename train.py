import os
import numpy as np
import argparse
from visualdl import LogWriter

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.optimizer import AdamOptimizer

from utils.utils import AverageMeter
from net.idcgan import Generator,Discriminator,Invertor
from loss.basic_seg_loss import Basic_Loss
from preprocessing import Normalize2
from dataloader import DataLoader, Transform

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='basic')
parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--epoch_num', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--image_folder', type=str, default='./dummy_data')
parser.add_argument('--image_list_file', type=str, default='./dummy_data/fabric_list.txt')
parser.add_argument('--checkpoint_folder', type=str, default='./output')
parser.add_argument('--save_freq', type=int, default=10)
args = parser.parse_args()

def main():
    # Step 0: preparation
    writer = LogWriter(logdir="./log/scalar")
    place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        # Step 1: Define training dataloader
        image_folder=""
        image_list_file="dummy_data/fabric_list.txt"
        transform = Transform() #Normalize2()  # [0,255]-->[0,1]
        x_data = DataLoader(image_folder,image_list_file,transform=transform)
        x_dataloader = fluid.io.DataLoader.from_generator(capacity=2,return_list=True)
        x_dataloader.set_sample_generator(x_data,args.batch_size)

        total_batch = len(x_data)//args.batch_size
        
        # Step 2: Create model
        if args.net == "basic":
            D = Discriminator()
            G = Generator()
            E = Invertor()
        else:
            raise NotImplementedError(f"args.net: {args.net} is not Supported!")

        # Step 3: Define criterion and optimizer
        criterion = Basic_Loss
        D_optim = AdamOptimizer(learning_rate=args.lr,parameter_list=D.parameters())
        G_optim = AdamOptimizer(learning_rate=args.lr,parameter_list=G.parameters())
        E_optim = AdamOptimizer(learning_rate=args.lr,parameter_list=E.parameters())

        G_loss_meter = AverageMeter()
        D_loss_meter = AverageMeter()
        E_loss_meter = AverageMeter()

        D.train()
        G.train()
        E.train()

        # Step 4: Slight Training
        iteration = -1
        is_slight_Train = True
        for epoch in range(1,args.epoch_num+1):
            #optim Discriminator
            for (x,x_labels) in x_dataloader():
                n = x.shape[0]
                if is_slight_Train:
                    iteration += 1
                    x = fluid.layers.cast(x,dtype="float32")
                    x = fluid.layers.transpose(x,perm=[0,3,1,2])
                    preds_x = D(x)
                    preds_x_array = preds_x.numpy()
                    #print("D(x),1",preds_array.shape, np.mean(preds_array))
                    writer.add_scalar(tag="D(x)=1", step=iteration, value=np.mean(preds_x_array))
                    if np.mean(preds_x_array)>=0.98:
                        is_slight_Train = False

                    z = np.random.rand(n,64)
                    zeros = np.zeros((n,1))
                    z = to_variable(z)
                    zeros = to_variable(zeros)
                    z = fluid.layers.cast(z,dtype="float32")
                    zeros = fluid.layers.cast(zeros,dtype="int64")
                    preds_fx = D(G(z))
                    preds_fx_array = preds_fx.numpy()
                    writer.add_scalar(tag="D(G(z))=0", step=iteration, value=np.mean(preds_fx_array))
                    D_loss = criterion(preds_x,x_labels) + criterion(preds_fx,zeros)
                    D_loss.backward()
                    D_optim.minimize(D_loss)
                    D.clear_gradients()
                    D_loss_meter.update(D_loss.numpy()[0], n)
                    writer.add_scalar(tag="D_loss", step=iteration, value=D_loss_meter.avg)
                    print(f"EPOCH[{epoch:03d}/{args.epoch_num:03d}], " +
                            f"STEP{iteration}, " +
                            f"Average D Loss: {D_loss_meter.avg:4f}, " )

                    z = np.random.rand(n,64)
                    ones = np.ones((n,1))
                    z = to_variable(z)
                    ones = to_variable(ones)
                    z = fluid.layers.cast(z,dtype="float32")
                    ones = fluid.layers.cast(ones,dtype="int64")
                    preds = D(G(z))
                    preds_array = preds.numpy()
                    writer.add_scalar(tag="D(G(z))=1", step=iteration, value=np.mean(preds_array))
                    G_loss = criterion(preds,ones)
                    G_loss.backward()
                    G_optim.minimize(G_loss)
                    G.clear_gradients()
                    G_loss_meter.update(G_loss.numpy()[0], n)
                    writer.add_scalar(tag="G_loss", step=iteration, value=G_loss_meter.avg)
                    print(f"EPOCH[{epoch:03d}/{args.epoch_num:03d}], " +
                            f"STEP{iteration}, " +
                            f"Average G Loss: {G_loss_meter.avg:4f}" )
                    
            if epoch % args.save_freq == 0 or epoch == args.epoch_num or not is_slight_Train:
                D_model_path = os.path.join(args.checkpoint_folder, f"D_{args.net}-Epoch-{epoch}")
                G_model_path = os.path.join(args.checkpoint_folder, f"G_{args.net}-Epoch-{epoch}")

                # save model and optmizer states
                model_dict = D.state_dict()
                fluid.save_dygraph(model_dict,D_model_path)
                optim_dict = D_optim.state_dict()
                fluid.save_dygraph(optim_dict,D_model_path)

                model_dict = G.state_dict()
                fluid.save_dygraph(model_dict,G_model_path)
                optim_dict = G_optim.state_dict()
                fluid.save_dygraph(optim_dict,G_model_path)

                print(f'----- Save model: {D_model_path}.pdparams, {G_model_path}.pdparams')
                if not is_slight_Train:
                    break;

        # Step 5:  full training for Generator and Discriminator
        D_optim = AdamOptimizer(learning_rate=args.lr*10,parameter_list=D.parameters())
        G_optim = AdamOptimizer(learning_rate=args.lr*10,parameter_list=G.parameters())
        G_loss_meter = AverageMeter()
        D_loss_meter = AverageMeter()

        for epoch in range(1,args.epoch_num+1):
            for (x,x_labels) in x_dataloader():
                n = x.shape[0]
                iteration += 1
                x = fluid.layers.cast(x,dtype="float32")
                x = fluid.layers.transpose(x,perm=[0,3,1,2])
                preds1 = D(x)
                preds_array = preds1.numpy()
                writer.add_scalar(tag="D(x)=1", step=iteration, value=np.mean(preds_array))
                z = np.random.rand(n,64)
                zeros = np.zeros((n,1))
                z = to_variable(z)
                zeros = to_variable(zeros)
                z = fluid.layers.cast(z,dtype="float32")
                zeros = fluid.layers.cast(zeros,dtype="int64")
                preds2 = D(G(z))
                preds_array = preds2.numpy()
                #print("DG(z),0:",preds_array.shape, np.mean(preds_array))
                writer.add_scalar(tag="D(G(z))=0", step=iteration, value=np.mean(preds_array))
                D_loss = criterion(preds1,x_labels) + criterion(preds2,zeros)
                D_loss.backward()
                D_optim.minimize(D_loss)
                D.clear_gradients()
                D_loss_meter.update(D_loss.numpy()[0], n)
                writer.add_scalar(tag="D_loss", step=iteration, value=D_loss_meter.avg)
                print(f"EPOCH[{epoch:03d}/{args.epoch_num:03d}], " +
                        f"STEP{iteration}, " +
                        f"Average D Loss: {D_loss_meter.avg:4f} " )
                z = np.random.rand(n,64)
                ones = np.ones((n,1))
                z = to_variable(z)
                ones = to_variable(ones)
                z = fluid.layers.cast(z,dtype="float32")
                ones = fluid.layers.cast(ones,dtype="int64")
                preds = D(G(z))
                preds_array = preds.numpy()
                #print("DG(z),1:",preds_array.shape, np.mean(preds_array))
                writer.add_scalar(tag="D(G(z))=1", step=iteration, value=np.mean(preds_array))
                G_loss = criterion(preds,ones)
                G_loss.backward()
                G_optim.minimize(G_loss)
                G.clear_gradients()
                G_loss_meter.update(G_loss.numpy()[0], n)
                writer.add_scalar(tag="G_loss", step=iteration, value=G_loss_meter.avg)
                print(f"EPOCH[{epoch:03d}/{args.epoch_num:03d}], " +
                        f"STEP{iteration}, " +
                        f"Average G Loss: {G_loss_meter.avg:4f}" )

            if epoch % args.save_freq == 0 or epoch == args.epoch_num:
                D_model_path = os.path.join(args.checkpoint_folder, f"D_{args.net}-Epoch-{epoch}")
                G_model_path = os.path.join(args.checkpoint_folder, f"G_{args.net}-Epoch-{epoch}")

                # save model and optmizer states
                model_dict = D.state_dict()
                fluid.save_dygraph(model_dict,D_model_path)
                optim_dict = D_optim.state_dict()
                fluid.save_dygraph(optim_dict,D_model_path)

                model_dict = G.state_dict()
                fluid.save_dygraph(model_dict,G_model_path)
                optim_dict = G_optim.state_dict()
                fluid.save_dygraph(optim_dict,G_model_path)
                print(f'----- Save model: {D_model_path}.pdparams, {G_model_path}.pdparams')

        # Step 6: full training for Inverter
        E_optim = AdamOptimizer(learning_rate=args.lr*10,parameter_list=E.parameters())
        E_loss_meter = AverageMeter()

        for epoch in range(1,args.epoch_num+1):
            for (x,x_labels) in x_dataloader():
                n = x.shape[0]
                iteration += 1
                x = fluid.layers.cast(x,dtype="float32")
                image = x.numpy()[0]*255
                writer.add_image(tag="x", step=iteration, img=image)
                x = fluid.layers.transpose(x,perm=[0,3,1,2])
                invert_x = G(E(x))
                invert_image = fluid.layers.transpose(invert_x,perm=[0,2,3,1])
                invert_image = invert_image.numpy()[0]*255
                #print("D(x),1",preds_array.shape, np.mean(preds_array))
                writer.add_image(tag="invert_x", step=iteration, img=invert_image)
                print(np.max(invert_image),np.min(invert_image))
                E_loss = fluid.layers.mse_loss(invert_x,x)
                print("E_loss shape:",E_loss.numpy().shape)
                E_loss.backward()
                E_optim.minimize(E_loss)
                E.clear_gradients()
                E_loss_meter.update(E_loss.numpy()[0], n)
                writer.add_scalar(tag="E_loss", step=iteration, value=E_loss_meter.avg)
                print(f"EPOCH[{epoch:03d}/{args.epoch_num:03d}], " +
                        f"STEP{iteration}, " +
                        f"Average E Loss: {E_loss_meter.avg:4f}, " )

            if epoch % args.save_freq == 0 or epoch == args.epoch_num:
                E_model_path = os.path.join(args.checkpoint_folder, f"E_{args.net}-Epoch-{epoch}")
                # save model and optmizer states
                model_dict = E.state_dict()
                fluid.save_dygraph(model_dict,E_model_path)
                optim_dict = E_optim.state_dict()
                fluid.save_dygraph(optim_dict,E_model_path)
                print(f'----- Save model: {E_model_path}.pdparams, {E_model_path}.pdparams')
                

if __name__ == "__main__":
    main()
