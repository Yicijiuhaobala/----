## Motivation
- 一般来说，提升网络性能最直接的办法就是增加网络深度和宽度，这也就意味着巨量的参数。但是巨量的参数容易产生过拟合也会大大增加计算量。
- 解决以上方法的根本方法是**将全连接甚至一般的卷积都转化为稀疏连接**。对于大规模稀疏的神经网络，可以通过分析激活值的统计特性和对高度相关的输出进行聚类来逐层构建出一个最有网路，这点表明臃肿的稀疏网络可能不失性能地简化。其实在早期，为了打破网络对称性和提高学习能力，传统网络都使用了随机稀疏连接，但**计算机软硬件对非均匀稀疏数据计算效率很差**。
- Inception 既能保持网络结构的稀疏性，又能利用密集矩阵的高计算性能。**将稀疏矩阵聚类为密集的子矩阵来提高计算性能**

### Inception模块
- 采用不同大小的卷积核意味着不同大小的感受野，最后拼接意味着不同尺度特征的融合
- 之所以卷积核采用1、3和5，主要是为了方便对齐。设定卷积步长stride=1之后，只要分别设定pad=0,1,2 那么卷积之后便可以得到相同维度的特征，然后这些特征就可以直接拼接在一起了
- 网络越到后面，特征越抽象，而且每个特征所涉及的感受野也越大，随着层数的增加，3*3和5*5卷积的比例也要增加。5*5的卷积核会带来巨大的运算量，采用1*1进行卷积核进行降维。
- 虽然移除了全连接层，但网络中依然使用了dropout层
- 为了避免梯度消失，网络额外增加了2个辅助的softmax用于向前传导梯度
- 准则： 
  - 避免表达瓶颈，特别是在网络靠前的地方。信息流前向传播过程中现任不能经过高度压缩的层，即表达瓶颈。从input 到output feature map宽和高基本都会逐渐变小，但是不能一下子就变得很小。输出的维度channel一般来说会逐渐增多，否则网络很难训练
  - 高纬度特征更易区分，会加快训练
  - 可以在低维嵌入上进行空间汇聚而无需担心丢失很多的信息。比如在进行3*3卷积之前，可以先对输入进行降维而不会产生严重的后果。假设信息可以被简单压缩，那么训练就会加快
  - 平衡网络的深度和宽度和维度

 ### 模型代码：
 ```python
    import torch.nn as nn
    import torch
    import torch.nn.functional as F


    class GoogLeNet(nn.Module):
        def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
            super(GoogLeNet, self).__init__()
            self.aux_logits = aux_logits

            self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

            self.conv2 = BasicConv2d(64, 64, kernel_size=1)
            self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
            self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

            self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
            self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
            self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

            self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
            self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
            self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
            self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
            self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
            self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

            self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
            self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

            if self.aux_logits:
                self.aux1 = InceptionAux(512, num_classes)
                self.aux2 = InceptionAux(528, num_classes)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.4)
            self.fc = nn.Linear(1024, num_classes)
            if init_weights:
                self._initialize_weights()

        def forward(self, x):
            # N x 3 x 224 x 224
            x = self.conv1(x)
            # N x 64 x 112 x 112
            x = self.maxpool1(x)
            # N x 64 x 56 x 56
            x = self.conv2(x)
            # N x 64 x 56 x 56
            x = self.conv3(x)
            # N x 192 x 56 x 56
            x = self.maxpool2(x)

            # N x 192 x 28 x 28
            x = self.inception3a(x)
            # N x 256 x 28 x 28
            x = self.inception3b(x)
            # N x 480 x 28 x 28
            x = self.maxpool3(x)
            # N x 480 x 14 x 14
            x = self.inception4a(x)
            # N x 512 x 14 x 14
            if self.training and self.aux_logits:    # eval model lose this layer
                aux1 = self.aux1(x)

            x = self.inception4b(x)
            # N x 512 x 14 x 14
            x = self.inception4c(x)
            # N x 512 x 14 x 14
            x = self.inception4d(x)
            # N x 528 x 14 x 14
            if self.training and self.aux_logits:    # eval model lose this layer
                aux2 = self.aux2(x)

            x = self.inception4e(x)
            # N x 832 x 14 x 14
            x = self.maxpool4(x)
            # N x 832 x 7 x 7
            x = self.inception5a(x)
            # N x 832 x 7 x 7
            x = self.inception5b(x)
            # N x 1024 x 7 x 7

            x = self.avgpool(x)
            # N x 1024 x 1 x 1
            x = torch.flatten(x, 1)
            # N x 1024
            x = self.dropout(x)
            x = self.fc(x)
            # N x 1000 (num_classes)
            if self.training and self.aux_logits:   # eval model lose this layer
                return x, aux2, aux1
            return x

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


    class Inception(nn.Module):
        def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
            super(Inception, self).__init__()

            self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

            self.branch2 = nn.Sequential(
                BasicConv2d(in_channels, ch3x3red, kernel_size=1),
                BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
            )

            self.branch3 = nn.Sequential(
                BasicConv2d(in_channels, ch5x5red, kernel_size=1),
                # 在官方的实现中，其实是3x3的kernel并不是5x5，这里我也懒得改了，具体可以参考下面的issue
                # Please see https://github.com/pytorch/vision/issues/906 for details.
                BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
            )

            self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                BasicConv2d(in_channels, pool_proj, kernel_size=1)
            )

        def forward(self, x):
            branch1 = self.branch1(x)
            branch2 = self.branch2(x)
            branch3 = self.branch3(x)
            branch4 = self.branch4(x)

            outputs = [branch1, branch2, branch3, branch4]
            return torch.cat(outputs, 1)


    class InceptionAux(nn.Module):
        def __init__(self, in_channels, num_classes):
            super(InceptionAux, self).__init__()
            self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
            self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

            self.fc1 = nn.Linear(2048, 1024)
            self.fc2 = nn.Linear(1024, num_classes)

        def forward(self, x):
            # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
            x = self.averagePool(x)
            # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
            x = self.conv(x)
            # N x 128 x 4 x 4
            x = torch.flatten(x, 1)
            x = F.dropout(x, 0.5, training=self.training)
            # N x 2048
            x = F.relu(self.fc1(x), inplace=True)
            x = F.dropout(x, 0.5, training=self.training)
            # N x 1024
            x = self.fc2(x)
            # N x num_classes
            return x


    class BasicConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, **kwargs):
            super(BasicConv2d, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x
