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
from dataloader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='basic')
parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--image_folder', type=str, default='./dummy_data')
parser.add_argument('--image_list_file', type=str, default='./dummy_data/fabric_list.txt')
parser.add_argument('--checkpoint_folder', type=str, default='./output')
parser.add_argument('--save_freq', type=int, default=10)
args = parser.parse_args()


def main():
    # Step 0: preparation

    # 在`./log/scalar_test/train`路径下建立日志文件
    writer = LogWriter(logdir="./log/scalar")
    # 使用scalar组件记录一个标量数据
    # writer.add_scalar(tag="acc", step=1, value=0.5678)
    # writer.add_scalar(tag="acc", step=2, value=0.6878)
    # writer.add_scalar(tag="acc", step=3, value=0.9878)

    place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        # Step 1: Define training dataloader
        image_folder=""
        image_list_file="dummy_data/fabric_list.txt"
        transform = None # Normalize2()
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
        # create optimizer
        D_optim = AdamOptimizer(learning_rate=args.lr,parameter_list=D.parameters())
        G_optim = AdamOptimizer(learning_rate=args.lr,parameter_list=G.parameters())
        E_optim = AdamOptimizer(learning_rate=args.lr,parameter_list=E.parameters())

        G_loss_meter = AverageMeter()
        D_loss_meter = AverageMeter()

        D.train()
        G.train()
        # Step 4: Training
        iteration = -1
        D_continue,G_continue =True,True
        for epoch in range(args.epoch_num):
            #optim Discriminator
            for (x,x_labels) in x_dataloader():
                n = x.shape[0]
                iteration += 1
                if D_continue:
                    x = fluid.layers.cast(x,dtype="float32")
                    x = fluid.layers.transpose(x,perm=[0,3,1,2])
                    preds1 = D(x)
                    preds_array = preds1.numpy()
                    #print("D(x),1",preds_array.shape, np.mean(preds_array))
                    writer.add_scalar(tag="D(x)=1", step=iteration, value=np.mean(preds_array))
                    if np.mean(preds_array)>=0.98:
                        D_continue = False

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
                            f"Average D Loss: {D_loss_meter.avg:4f}, " )

                if G_continue:
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

            # print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss:.4f}")

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



if __name__ == "__main__":
    main()
