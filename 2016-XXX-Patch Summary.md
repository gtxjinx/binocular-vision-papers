概述
====================================
&emsp;&emsp;这篇论文提出了一个新的预测立体图像匹配的准确性的方法，这个方法被称为confidence。这个方法的输入包含两个通道，一通道的方法来源于，相邻的像素如果具有一致的视差，则这种匹配更可能正确。二通道的方法是，对左图像做视差和对右图像做视差应该具有相同的结果。CNN的输入是disparity patch，这样features和classifier就可以同时训练。此外，confidence在直接调整参数后卡可以和SGM融合。这个方法在KITTI上表现很好。

相关知识
========================================
- probability mass function(概率质量函数)：与概率密度函数向对应，是离散的随机变量在各个点上的概率值。
- patch：指一组相邻的像素点集合。
- SGM semi global
-matching(半全局匹配算法)：能量函数如下：
```math
E(D)=\sum_x(C(x,D_x)+\sum_{y\in N_x}P_1T[|D_x-D_y|=1]+\sum_{y\in N_x}P_2T[|D_x-D_y|>1])
```
&emsp;&emsp;其中，D代表视差图，x，y代表像素点，`$ N_x$`指和x相邻的像素点（一般为8连通），`$ C(x,D_x) $`指x点视差为`$ D_x $`时的cost，P1、P2均为惩罚系数，P1代表相邻像素与x视差相差1时的惩罚系数，P2代表相邻像素与x视差相差大于1时的惩罚系数。`$ T[.] $`代表克罗内克函数（Kronecker delta），.为真时输出1，否则输出0.

- cost volume ：对于图像中每一个像素，生成该像素从视差为0到视差为max的所有匹配代价，构为cost volume

主要方法
============================================
1. discriminative information:有很多特征都可以用来预测confidence。Hu和Mordohai把他们分为五组。第一组是对匹配代价的考量，如果匹配代价过大，则不太可能是正确的匹配。第二组特征抓住了代价曲线的特性，例如无纹理的区域，曲线对应的值越小，则不确定性就越大。第三组特征是基于cost curve的局部最小值，比如PKR（Peak Ratio）。PKR等于最小匹配代价和第二小的局部最小值之比。第四组特征，是用整个cost curve去计算视差的概率质量函数。第五组特征，是对左右图像视差一致性的考察。

&emsp;&emsp;我们首先对输入进行选取，图相对、视差的cost volume 和视差图中更好的将作为输入。
最终，视差图上的patch表现较好。

2. confidence estimation with a CNN：考虑MED方法（Difference with Median Disparity），这个方法提出相邻的像素，如果视差一致，则更有可能是正确的匹配。把一个patch上的视差值均减去该patch中心的视差值`$ x_c $`，得到`$ p_1 $`.即
```math
\mathbf p_1=[D_1(x)-D_1(x_c)]_{x\in W}
```
把另一幅图像上与之相对应的patch，进行如下转换，得到
```math
\mathbf p_2=[D_2(x)-D_1(x_c)]_{x\in W}
```
&emsp;&emsp;我们把数据对`$ p=(p_1,p_2) $`作为CNN的输入。CNN作为分类器来训练，对于每一个patch，都有一个标签与之对应，表示这个patch的中心像素的对应是否正确。具体的CNN网络为，p为15\*15的输入，第一层包括六个3\*3卷积核和RELU，第二、三、四层每一层分别有4个不同的3\*3的卷积核和RELU。第五层是一个全连接层，有两个输出，这两个输出连接到一个sofmax层上，输出最终的结果。

&emsp;&emsp;需要注意的是，这种输入方法，由于每次只能输出一个patch中心点的confidence，所以每次计算都需要从头开始，很繁琐，我们把这种patch称为“disposable patch”所以定义一种与中心点无关的patch，这样可以通过传播整个图像一次计算出所有的confidence，我们把它叫做“reusable patch”，`$ p'=(p_1',p_2') $`,定义如下：
```math
\mathbf p_1'=[D_1(x)]_{x\in W} 

\mathbf p_2'=[D_2(x)-D_1(x)]_{x\in W} 
```

&emsp;&emsp;为了减少计算时间，我们开发了一个混合网络。网络分为两部分，第一部分把reusbale patch输入 confidence prediction network 中。第二部分把一个3\*3的disposable patch输入另一个CNN中，这个CNN第一层有三个不同的3\*3的卷积核和RELU，第二层有一个六个输出的全连接层。然后，把两部分的输出相结合，传入一个有两个输出的全连接层，再传入一个softmax层，输出最后的prediction。这个混合网络比普通网络效果更好。

3. confidence fusion for dense disparity map:我们假设cost volume上不连续的点，都有较大的图像梯度，这点和SGM的假设相同。但是，并不是所有的大梯度的点都是不连续的。我们认为，有高confidence的点，应该容易成为不连续点。因此，对高confidence像素的惩罚因数应该降低。我们这样定义`$ p_1 $`和`$ p_2 $`：
```math
P_{1,2}(x)=f_{1,2}(I(x))+P'_{1,2}\lambda_{max}(-\xi(x)+m,0)
```
其中`$ \xi(x) $`代表像素x的confidence，并且被归一化到0~1。`$ m $`和`$ \lambda $`是参数,范围在0~1之间。`$ P'_{1,2}$`是每个惩罚系数的最大值。

理解
=========================
&emsp;&emsp;这篇论文主要介绍了一种匹配置信度的预测算法。在训练过程中，通过采用一个hybrid network，把disposable patch和reusable patch相结合，达到准确率和计算时间的全面优化。之后，把这种confidence prediction与SGM相结合，还可以得到一个对dense disparity的confidence估计。