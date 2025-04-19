import torch.nn as nn
import torch
from module.swin_transformer_Har import *

def weights_init_normal(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

class DenseBlock(nn.Module):
    def __init__(self, in_channel, k, num_module=4):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_module):
            layer.append(self.conv_block(
                k * i + in_channel, k))
        self.net = nn.Sequential( * layer)

    def conv_block(self, input_channels, k):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.LeakyReLU(),
            nn.Conv2d(input_channels, k, kernel_size=3, padding=1))
            
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)# 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim = 1)
        return X

class Conv_path(nn.Module):
    def __init__(self, in_channel=32,k=8):
        super(Conv_path, self).__init__()
        
        self.Dense = DenseBlock(in_channel=in_channel, k=k)
        
        self.final_conv = nn.Conv2d(4*k+ in_channel, 32, 1)

    def forward(self, x):
        
        
        x1 = self.Dense(x)     
        
        x2 = self.final_conv(x1)

        return x2

class JCT_Hardanger(nn.Module):
    def __init__(self, in_channel=1, shallow_dim=32, num_layers=4):
        super(JCT_Hardanger,self).__init__()
      
        self.shallow_feature = nn.Sequential(nn.Conv2d(in_channel, shallow_dim, 3, 1, 1),
                                        nn.BatchNorm2d(shallow_dim), nn.LeakyReLU())     #  padding = (kernel_size - 1) // 2        

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = Conv_path()
            self.layers.append(layer)
        
        self.swin = SwinTransformer()
        #########   out Layer   ########## 
        self.out_layer = nn.Sequential(nn.BatchNorm2d(shallow_dim), nn.LeakyReLU(),nn.Conv2d(shallow_dim, in_channel, 3, 1, 1))

    

    def forward(self, x):
      
        x1 = self.shallow_feature(x)

        for layer in self.layers:
            x1 = layer(x1)

        x2 = self.swin(x1)

        out = self.out_layer(x2)

        mask = torch.zeros_like(x)
        mask[x == 0] = 1
        output = torch.mul(mask, out) + x
           
        return output


if __name__ == "__main__":
    
    X = torch.rand(size=(32, 1, 16, 1024))
    net = JCT_Hardanger()
    out = net(X)
    print(out.shape)