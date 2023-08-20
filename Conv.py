import torch
import torch.nn as nn

# Group convolution  组卷积
class GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super(GroupConv, self).__init__()
        # 把输入输出分组
        self.split_in = nn.Conv2d(in_channels, in_channels * groups,
                                  kernel_size=1, groups=in_channels)
        self.split_out = nn.Conv2d(in_channels * groups, out_channels,
                                   kernel_size=1, groups=groups)
        # 进行组卷积
        self.group_conv = nn.Conv2d(in_channels * groups,
                                    in_channels * groups,
                                    kernel_size=kernel_size,
                                    groups=in_channels * groups)

    def forward(self, x):
        # Split the input channels
        x = self.split_in(x)
        # Apply the group convolution
        x = self.group_conv(x)
        # Split the output channels
        x = self.split_out(x)
        return x

# Pyramid convolution 金字塔卷积
class PyramidConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PyramidConv, self).__init__()
        # Define three different kernels with different sizes and strides
        self.kernel_1 = nn.Conv2d(in_channels // 3,
                                  out_channels // 3,
                                  kernel_size=3,
                                  stride=1)
        
        self.kernel_2 = nn.Conv2d(in_channels // 3,
                                  out_channels // 3,
                                  kernel_size=5,
                                  stride=1,
                                  padding=1)

        
        self.kernel_3 = nn.Conv2d(in_channels // 3,
                                  out_channels // 3,
                                  kernel_size=7,
                                  stride=1, 
                                  padding=2)

    def forward(self,x):
       # Split the input channels into three parts
       x1,x2,x3=torch.chunk(x,dim=1,chunks=3) 
       # Apply the different kernels on each part
       x1=self.kernel_1(x1) 
       x2=self.kernel_2(x2) 
       x3=self.kernel_3(x3) 
       print(x1.shape, x2.shape, x3.shape)
       # Concatenate the outputs along the channel dimension
       x=torch.cat([x1,x2,x3],dim=1) 
       return x
if __name__ == "__main__":
    model1 = GroupConv(in_channels=513, out_channels=256, kernel_size=3, groups=8)
    model2 = PyramidConv(in_channels=513, out_channels=256)
    x = torch.randn((1, 513, 256, 256))
    y1 = model1(x)
    y2 = model2(x)
    print(y1)
    print(y2)