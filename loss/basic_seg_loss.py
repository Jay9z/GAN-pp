import paddle
import paddle.fluid as fluid
import numpy as np
import cv2

eps = 1e-8

def Basic_Loss(preds, labels):
    n, c = preds.shape
    # create cross_entropy criterion

    # transpose preds to NxC
    #preds = fluid.layers.transpose(preds,(0,2,3,1))
    # print(type(preds),type(labels))
    preds = fluid.layers.cast(preds,dtype="float32")
    labels = fluid.layers.cast(labels,dtype="float32")
    loss = fluid.layers.cross_entropy(input=preds,label=labels,soft_label=True) +\
     fluid.layers.cross_entropy(input=(1-preds),label=(1-labels),soft_label=True)
    
    avg_loss = fluid.layers.mean(loss) 

    return avg_loss

def Basic_SegLoss(preds, labels, ignore_index=255):
    n, c, h, w = preds.shape

    # create softmax_with_cross_entropy criterion
    # transpose preds to NxHxWxC
    preds = fluid.layers.transpose(preds,(0,2,3,1))
    loss = fluid.layers.softmax_with_cross_entropy(preds,labels)
    
    mask = labels!=ignore_index
    mask = fluid.layers.cast(mask, 'float32')

    # call criterion and compute loss
    
    loss = loss * mask
    avg_loss = fluid.layers.mean(loss) / (fluid.layers.mean(mask) + eps)

    return avg_loss

def main():
    label = cv2.imread('dummy_data/GroundTruth_trainval_png/2008_000026.png')
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY).astype(np.int64)
    pred = np.random.uniform(0, 1, (1, 59, label.shape[0], label.shape[1])).astype(np.float32)
    label = label[:,:,np.newaxis]
    label = label[np.newaxis, :, :, :]

    with fluid.dygraph.guard(fluid.CPUPlace()):
        pred = fluid.dygraph.to_variable(pred)
        label = fluid.dygraph.to_variable(label)
        loss = Basic_SegLoss(pred, label)
        print(loss.numpy())

if __name__ == "__main__":
    main()

