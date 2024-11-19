import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight.data)

class DenseBlock(nn.Module):
    def __init__(self, in_channel, k, num_module=4):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_module):
            layer.append(self.conv_block(
                k * i + in_channel, k))
        self.net = nn.Sequential( * layer)
        self.act = nn.ReLU()
        
    def conv_block(self, input_channels, k):
       return nn.Sequential(
           nn.Conv1d(input_channels, k, kernel_size=3, padding=1))
            
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)# 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim = 1)
        out = self.act(X)
        return out


class TD_layer(nn.Module):
    def __init__(self, in_channel):
        super(TD_layer,self).__init__()
        self.conv=nn.Conv1d(in_channel, in_channel, kernel_size=2, stride=2)
        
    def forward(self, x1):             
        out = self.conv(x1)               
        return out


class TU_layer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TU_layer,self).__init__()
        self.conv=nn.ConvTranspose1d(in_channel, out_channel, kernel_size=2, stride=2)
        self.act = nn.ReLU()
        
    def forward(self, x1):             
        out = self.conv(x1)      
        out = self.act(out) 
        return out


class skip(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(skip,self).__init__()
        self.connc=nn.Conv1d(in_channel, out_channel, kernel_size=1)
        self.act = nn.ReLU()
        
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)             
        out = self.connc(x)  
        out = self.act(out)
        return out

class decoder1(nn.Module):
    def __init__(self):
        super(decoder1, self).__init__()

        self.skip2 = skip(347, 87)
        self.DenseBD2 = DenseBlock(in_channel=87, k=24)
        self.TU1 = TU_layer(in_channel=183, out_channel=92)
        self.skip1 = skip(236,59)
        self.DenseBD1 = DenseBlock(in_channel=59, k=24)
        self.final = nn.Conv1d(155, 1, 1, 1)

    def forward(self, x , y,  z):
        x1 = self.skip2(x, y)
        x2 = self.DenseBD2(x1)
        x3 = self.TU1(x2)
        x4 = self.skip1(x3, z)
        x5 = self.DenseBD1(x4)
        out = self.final(x5)
        return out

class MultiBranchDecoder(nn.Module):
    def __init__(self, num_decoder=20):
        super(MultiBranchDecoder, self).__init__()
        
        self.task_decoders = nn.ModuleList()
        for _ in range(num_decoder):
            decoder = decoder1()
            self.task_decoders.append(decoder)
    
    def forward(self, x, y, z):
        
        task_outputs = []
        for decoder in self.task_decoders:
            task_output = decoder(x, y, z)
            task_outputs.append(task_output)
            out = torch.cat(task_outputs, dim=1)       
        return out


    
class DFCN(nn.Module):
    def __init__(self, in_channel=20, out_channel=20):
        super(DFCN,self).__init__()
        
        #########   first conv layer   ##########
        self.first_conv = nn.Sequential(nn.Conv1d(in_channel, 48, kernel_size=3, stride=1, padding = (3 - 1) // 2), nn.ReLU()  )   #  padding = (kernel_size - 1) // 2
        
        ##########   down sampling   ############
        self.DenseBD1 = DenseBlock(in_channel=48, k=24)
        self.DenseBD2 = DenseBlock(in_channel=144, k=24)
        self.DenseBD3 = DenseBlock(in_channel=240, k=24)
        self.DenseBD4 = DenseBlock(in_channel=336, k=24)     
        self.TD1 = TD_layer(144)
        self.TD2 = TD_layer(240)
        self.TD3 = TD_layer(336)
        self.TD4 = TD_layer(432)
        
        ##########   Bottleneck   ############
        self.bottleneck = DenseBlock(in_channel=432, k=24)
        
        ##########   up sampling   ############
        self.TU4 = TU_layer(in_channel=528, out_channel=264)
        self.TU3 = TU_layer(in_channel=270, out_channel=135)
        self.TU2 = TU_layer(in_channel=214, out_channel=107)
        self.skip4 = skip(696, 174)
        self.skip3 = skip(471, 118)
        self.DenseBU4 = DenseBlock(in_channel=174, k=24)
        self.DenseBU3 = DenseBlock(in_channel=118, k=24)

        self.multidecoder = MultiBranchDecoder()

    def forward(self, x):
        
        #########   first conv layer   ##########
        x1 = self.first_conv(x)
        ##########   down sampling   ############
        x2 = self.DenseBD1(x1)
        # print(x2.shape)
        x3 = self.TD1(x2)
        x4 = self.DenseBD2(x3)
        x5 = self.TD2(x4)
        x6 = self.DenseBD3(x5)
        x7 = self.TD3(x6)
        x8 = self.DenseBD4(x7)
        x9 = self.TD4(x8)
        # ##########   Bottleneck   ############
        x10 = self.bottleneck(x9)
        # #########   up sampling   ############
        x11 = self.TU4(x10)
        x12 = self.skip4(x11, x8)
        x13 = self.DenseBU4(x12)
        x14 = self.TU3(x13)
        x15 = self.skip3(x14, x6)
        x16 = self.DenseBU3(x15)
        x17 = self.TU2(x16)
        x18 = self.multidecoder(x17, x4, x2)
        x19 = x18 + x
        
        # mask = torch.zeros_like(x)
        # mask[x == 0] = 1
        # output = torch.mul(mask, x19) + x
        return x19
 

if __name__ == "__main__":
    
    X = torch.rand(size=(32, 20, 1024))
    net = DFCN()
    out = net(X)
    print(X.shape)
    print(out.shape)

