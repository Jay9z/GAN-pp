import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D, Linear, Conv2DTranspose
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Dropout

import numpy as np
np.set_printoptions(precision=2)

class Generator(Layer):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1 = Linear(input_dim=64,output_dim=1024)
        self.bn1 = BatchNorm(num_channels=256,act='relu')
        self.conv2 = Conv2DTranspose(num_channels=256,num_filters=192,filter_size=3,padding=1,output_size=4,stride=2)
        self.bn2 = BatchNorm(num_channels=192,act='relu')
        self.conv3 = Conv2DTranspose(num_channels=192,num_filters=128,filter_size=3,padding=1,output_size=8,stride=2)
        self.bn3 = BatchNorm(num_channels=128,act='relu')
        self.conv4 = Conv2DTranspose(num_channels=128,num_filters=64,filter_size=3,padding=1,output_size=16,stride=2)
        self.bn4 = BatchNorm(num_channels=64,act='relu')
        self.conv5 = Conv2DTranspose(num_channels=64,num_filters=1,filter_size=3,padding=1,output_size=32,stride=2,act='tanh')

    def forward(self,inputs):
        x = self.fc1(inputs)
        x = fluid.layers.reshape(x,shape=(-1,256,2,2))
        x = self.bn1(x)
        x = self.conv2(x)
        #print(x.shape)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        return x

class Discriminator(Layer):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = Conv2D(num_channels=1,num_filters=64,filter_size=3,padding=1,stride=2,act='leaky_relu')
        self.conv2 = Conv2D(num_channels=64,num_filters=128,filter_size=3,padding=1,stride=2)
        self.bn2 = BatchNorm(num_channels=128,act='leaky_relu')
        self.conv3 = Conv2D(num_channels=128,num_filters=192,filter_size=3,padding=1,stride=2)
        self.bn3 = BatchNorm(num_channels=192,act='leaky_relu')
        self.conv4 = Conv2D(num_channels=192,num_filters=256,filter_size=3,padding=1,stride=2)
        self.bn4 = BatchNorm(num_channels=256,act='leaky_relu')
        self.fc = Linear(input_dim=1024,output_dim=1,act='sigmoid')
    
    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = fluid.layers.reshape(x,shape=(-1,1024))
        x = self.fc(x)
        return x

class Invertor(Layer):
    def __init__(self):
        super(Invertor,self).__init__()
        self.conv1 = Conv2D(num_channels=1,num_filters=64,filter_size=3,padding=1,stride=2,act='leaky_relu')
        self.conv2 = Conv2D(num_channels=64,num_filters=128,filter_size=3,padding=1,stride=2)
        self.bn2 = BatchNorm(num_channels=128,act='leaky_relu')
        self.conv3 = Conv2D(num_channels=128,num_filters=192,filter_size=3,padding=1,stride=2)
        self.bn3 = BatchNorm(num_channels=192,act='leaky_relu')
        self.conv4 = Conv2D(num_channels=192,num_filters=256,filter_size=3,padding=1,stride=2)
        self.bn4 = BatchNorm(num_channels=256,act='leaky_relu')
        self.fc = Linear(input_dim=1024,output_dim=64,act='tanh')
    
    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = fluid.layers.reshape(x,shape=(-1,1024))
        x = self.fc(x)
        return x


def main():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        z_data = np.random.rand(100,64).astype(np.float32)
        x_data = np.random.rand(100,1,32,32).astype(np.float32)
        print(z_data.shape)
        z = to_variable(z_data)
        x = to_variable(x_data)
        D = Discriminator()
        E = Invertor()
        G = Generator()
        G.train()
        D.train()
        E.train()
        g = G(z)
        d = D(x)
        e = E(x)
        print("g shape:",g.shape)
        print("d shape:",d.shape)
        print("e shape:",e.shape)

if __name__ =="__main__":
    main()