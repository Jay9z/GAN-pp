import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Pool2D#TODO
from paddle.fluid.dygraph import Conv2D#TODO
from paddle.fluid.dygraph import to_variable#TODO
import numpy as np
np.set_printoptions(precision=2)


class BasicModel(fluid.dygraph.Layer):
    # BasicModel contains:
    # 1. pool:   4x4 max pool op, with stride 4
    # 2. conv:   3x3 kernel size, takes RGB image as input and output num_classes channels,
    #            note that the feature map size should be the same
    # 3. upsample: upsample to input size
    #
    # TODOs:
    # 1. The model takes an random input tensor with shape (1, 3, 8, 8)
    # 2. The model outputs a tensor with same HxW size of the input, but C = num_classes
    # 3. Print out the model output in numpy format 

    def __init__(self, num_classes=59):
        super(BasicModel, self).__init__()
        self.conv = Conv2D(3,num_classes,3,padding=1)
        self.pool = Pool2D(pool_size=4,pool_type='max',pool_stride=4)

    def forward(self, inputs):
        x = self.pool(inputs)
        x = self.conv(x)
        x = fluid.layers.interpolate(x, out_shape=(inputs.shape[2], inputs.shape[3]))
        
        return x

def main():
    #place = paddle.fluid.CPUPlace()
    #place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard():
        model = BasicModel(num_classes=59)
        model.eval()
        input_data = np.random.rand(1,3,8,8).astype('float32')  
        input_data = to_variable(input_data) 
        print('Input data shape: ', input_data.shape)
        output_data = model(input_data) 
        output_data = output_data.numpy()
        print('Output data shape: ', output_data.shape)

if __name__ == "__main__":
    main()