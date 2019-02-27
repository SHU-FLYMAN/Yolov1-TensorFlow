[TOC]

> 网站地址：

这个系列，暂且叫做《**飞翔的荷兰人带你入门目标检测**：第一季》，是为了帮助更多的人入门目标检测的，第一季我们将讲完整个YOLO系列。

我们假设：

1. 你有一定的Tensorflow基础
2. 你有一定的Git的使用经验
3. 你对Deep learning有一定深入的了解(必备)

除了第三条，剩下的都可以一边学习一边补充。因为这个教程也是我自己的复现Yolov1的学习笔记。此前一直使用Keras做网络结构方面的研究，Git也没什么经验，开发过机器学习系统，效果还不错，部署到了阿里云上，拿过一个竞赛的第10名，但水平说实话吧确实堪忧。

----

说到 Yolov3，必须从v1说起，很多文章都没有讲清楚细节。想要真正弄明白就要从v1说起，因为很多实现细节等等论文v3中都没有提到过，更多的细节需要自己手敲一遍代码才能明白。

速成的事情很少，但凡事都有例外，我们这里就有个速成的Yolov3教程，包括代码跟实现。

第一季的第一期我们会讲的啰嗦一点，务必让零基础的同学能够跟着入门。当讲到后面几期的时候，我们会加快讲速度，也会给大家补充网络的训练技巧，比如如何使用预训练权重、如何分布式训练模型等等技巧。这里只是强行训练网络，效果并不会太好。

# 1 YOLOv1理论

> 论文标题: 《You Only Look Once: Unified, Real-Time Object Detection》
>
> 论文地址：https://arxiv.org/pdf/1506.02640.pdf
>
> 代码地址：

## 1.1 Yolov1初识

---

Yolov1是Yolo系列的开山之作，以简洁的网络结构，详细的复现过程（作者给出详细教程）而受到CVer们的追捧。

> **Yolov1奠定了Yolo系列算法“分而治之”，粗暴回归的基调。**

在Yolov1上，输入图片被划分为 $7\times7$ 的网格，如下图所示，每个单元格独立作检测：

![img](https://img-blog.csdn.net/20180910124838265?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xldmlvcGt1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

图片划分为 $S \times S $ 个 cell,经过网络提取特征后输出

- Bounding boxes $ x, y, w, h$(后面会修正)
- Confidence 置信度
- Class probability 单元格的类别概率

综合每个cell的这些信息做了极大值抑制最终实现物体目标检测，具体的细节会复杂一些。Yolo相比较其他目标检测算法，确实会难一点。但是后面很多One-stage都是在它基础上做的改进，如果要入门目标检测的话，我们不得不学习Yolo系列。

这里很容易被误导：

> 每个单元格的视野有限，而且很可能只有局部特征。这样就很难理解Yolo为何能检测到比grid_cell(网格单元)大很多的物体。

**其实Yolo的做法并不是把每个单独的网格作为输入feed到模型中，在Inference的过程中，网格只是起到物体中心点位置的划分之用，并不是对图片进行切片，不会让网格脱离整体的关系。**

## 1.2 Inference过程

具体的Inference做法如下：

**每个网格 cell 预测 B(B=2) 个boxes(X-center，Y-center，W，H，Confidence)和 类别class(20个类别)**

![img](https://img-blog.csdn.net/20180130221547656?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2h1MjAyMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

1. **boxes四个坐标**

   - x, y 为相对于到整个 cell 的左上角 的 0-1 范围的比例（基准为整个cell的大小）

     这里要说一下坐标系，不同于一般的数学坐标系，其$y​$轴方向向下，为$h​$的方向：

     ![](https://flyman-cjb.oss-cn-hangzhou.aliyuncs.com/picgos/20190223202154.png)

     **注意坐标转换x,y只是偏移量坐标，例如第2行第二列的grid的一个box坐标预测是0.2, 0.5，实际上坐标是(1+0.2, 1+0.5)/7归一化到原图尺寸的坐标(事实上做label时候的我们也是以全图坐标系为基准的)**

   - w, h 为相对于整个图像 0-1 范围的值

   - 对于落有object的cell，它的标签为上面说的值，否则则为空(计算机处理稀疏矩阵比较麻烦，暂且我们这样认为)。

2. **Confidence**

   Confidence作者的本意是想要预测出这个cell中是否存在object，正如做label时候，Confidence代表这个框内是否含落有某个object的中心，为0-1标签。

   但如果直接用Prob来回归，其实也还OK，但是总觉得缺少一点东西，整个网络只要预测出这个cell里是否有东西就OK了，跟预测出来的框和原始Object的重合度没有任何关系。

   > 举个例子，预测出来的Box离原来的object很远，但是仍旧在这个cell里面。在坐标损失中，预测Box离得很远会被惩罚，但在Confidence损失中，竟然没有受到任何惩罚。

   因此作者引入IOU，将两者相乘，如果你预测离得很偏，这个损失就会很大。预测出的这个代表所预测的Confidence因此含有两重信息，计算公式如下：
   $$
   Confidence = \Pr \left( {Object} \right) \times IOU_{pred}^{truth}
   $$

   - 整个cell存在object的概率
   - 预测出这个box有多准（用IOU替代）

   但记住，做label时候，Confidence标签只代表这个cell中是否有Object。

3. **类别**

   同样，如果这个cell的中心落有object，这个网格就会用one-hot标签标记出这个目标的类别，如果不含有目标，则为空。

   每个cell中讲道理是可能有多个类别的，但是Yolo认为整个cell里面只含有一个类别。因此每个cell只预测一组类别，跟该cell有多少个boxes无关。

   但是做标签的时候，也许一个cell中可能有2个类别，但这没关系，就让神经网络训练的时候，见到第一个标签时候认为整个cell属于某个类别，见到下一个标签，认为整个cell又属于另外个标签。反正这种情况并不多见，此外，Yolov3对此做了改进，就让我们暂且视而不见吧。

总结一下我们最终的输出大小：
$$
（ 7×7）.cells ×（（4.coords+1.confidence）×2.boxes+20.classes）= 7*7*30 = 1470\times1
$$
具体的网络结构：

![img](https://img-blog.csdn.net/20180910130225149?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xldmlvcGt1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

最后接入Loss层进行损失计算，最终反向传播梯度计算来强行predict出这些位置、置信度等，这是训练的过程。

> 最后的$7 \times 7 \times 30$，特征图上每个30维的特征和原来图像是有一一对应关系的，可以认为，其实一个cell的信息都包含在最后特征图的一个$7 \times 7 \times 30$的一个cell 的 Tensor中。

等到predict的时候，过程会有些不一样，后面会说。下面是整个网络的结构的代码(不包含损失层)（网络结构有些不一样，但没关系）：

```python
def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            # 设置默认参数
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha),
                weights_regularizer=slim.l2_regularizer(0.0005),
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
                # 对输入在宽度和高度上进行填充
                # input shape batchx448x448x3 >>> batchx454x454x3
                net = tf.pad(
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')
                # conv1 7x7x64 s=2
                # padding valid
                net = slim.conv2d(
                    net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                # conv2 3x3x64x192
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                # 一组卷积操作
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
 
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
 
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
 
                # pad batchx16x16x1024
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                net = slim.conv2d(
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')
 
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
 
                # outshape batchx7x7x1024
              # 转置，由[batch, image_height,image_width,channels]变成[bacth, channels, image_height,image_width]
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                # 转置后将按照每一个7x7的矩阵按每一行进行展开
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                net = slim.fully_connected(
                    net, num_outputs, activation_fn=None, scope='fc_36')
                # logists batchx(7x7x30) = batchx1470
        return net
```

我们网络的输出为$batch  \times 1470$ ，但后面计算损失的时候我们会将其`reshape`为$7 \times 7 \times 30$。

## 1.3 关于Inference的几个问题

之前我们说，Yolo首先把输入图像resize到 448 x 448，然后把输入图像划分成 $S\times S​$个grid，论文中是$7 \times 7​$，所以每个cell的Box的分辨率是$64 \times 64​$。还有网络最后的输出是$batch \times 1470​$，但有很多实现的细节我们是不理解的，要理解Yolo系列，必须弄明白这些细节。主要问题有：

1. 我们如何制作训练标签？

2. Yolo在把输入图像划分为$7 \times 7​$之后，图像中的一个目标的Box的中心点若是落入在某个网格内，则由这个网格负责预测这个目标(事实上只是起到坐标基准点的作用)。说起来容易，但我们怎么实现回归的时候让每个cell只负责自己的object？

3. 损失函数怎么计算？

   Yolo的损失函数相对来说还是比较复杂的，有坐标损失，Confidence损失、以及类别损失。这些损失如何综合起来计算？

4. 如何向量化处理整个运算？

至于如何predict，不要着急，让我们一个问题一个问题慢慢说来。

## 1.4 YOLO概括总结

为了方便大家继续下一部分内容，我们回顾一下下面几点：

1. 我们的网络输出为$batch  \times 1470​$ :

   为了便于矩阵运算提升计算效率，我们将计算矩阵化，我们网络的输出$batch  \times 1470​$，之后我们会将其`reshape`为$7 \times 7 \times 30​$来方便损失函数计算：

   ![img](https://img-blog.csdn.net/20180130221547656?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2h1MjAyMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

2. 我们 Inference 出的每个 cell 预测出有30维，其中

   - 前8维是回归box的坐标，后2维是box的Confidence，最后20维是类别(为了方便操作，我们将两个box的位置合并)
   - 坐标的 $x, y$ 用对应 **网格的左上角位置** 归一化到 **0-1** 之间的值
   - $w, h​$ 是对应 **整张图像** 的 width 和 height 归一化到 **0-1** 之间的值。
   - Confidence label为0-1标签，而 inference 的Confidence值为$Pr \times IOU$。
   - class 是整个cell 属于某个类别的概率。

3. 我们的label，每个grid 中label有25维度，只包含一个物体的信息。

   如果有多个，那也是分为多个label标签，一次只让神经网络见一个。这也是Yolov1精度不高的原因。

明白了这些，我们讲一下这个损失函数的设计。

## 1.5 损失函数构成 (0)

Yolo的损失函数包括三部分，分别是:

- 位置坐标损失 $x, y, w, h$
- 置信度损失 Confidence
- 类别预测损失 Classification

我们需要在这三部分损失函数之间找到一个平衡点。Yolo设计的损失函数如下(我将box分开了，方便大家看，实现的时候是取前8个为box坐标，后2个为Confidence，最后20个为整个cell的类别概率，现在看不懂没有关系，计算的时候我们将会说如何实现这个损失函数)：

![](https://flyman-cjb.oss-cn-hangzhou.aliyuncs.com/picgos/20190225104944.png)

## 1.6 损失函数设计考量上 (1)

我们如何在三部分损失函数之间找到平衡点？YOLO主要从以下几个方面考虑:

### 1.6.1 **坐标损失函数** 与 **类别损失函数**

> 每个小区域输出的8维位置坐标偏差的权重应该比20维类别预测偏差的权重要大。

1. 因为从体量上考虑，20维的影响很容易超过8维的影响，导致分类准确但位置偏差过大。

2. 再者，最后会在整体的分类预测结果上综合判断物体的类别（极大值抑制等）。因此单个小区域的分类误差稍微大一些，不至于影响最后的结果。

因此最终设置：

> **位置坐标损失** 和 **类别损失函数** 的权重比重为 **5 : 1**。

### 1.6.2 **置信度损失函数**

> 我们需要减弱不包含object的网格对网络的贡献。

在不包含有目标物体的网格中，cell的标签置信度为0。但是图片上大部分区域都是不含目标物体的(即使物体在整张图片上占比很大，由于Yolo的思想，其体现在标签上也仅仅是落有中心点的cell置信度为1)。

这些置信度为0的标签对梯度的贡献会远远大于含有目标物体置信度为1的网格对梯度的贡献，这就容易导致网络不稳定或者发散。

换句话说，网络会倾向于预测每个cell都不含有物体。因为大部分情况下，这种情况是对的。所以需要减弱不包含object的网格的贡献。

因此最终设置：

> 在**不包含 object** 的cell计算损失时，**置信度损失**的**权重为0.5**，在**包含 object时，权重取1**。

### 1.6.3 **目标大小有区别**

考虑到目标物体有大有小，对于大的物体，坐标预测存在一些偏差无伤大雅；但对于小的目标物体，偏差一点可能就是另外一样东西了。

因此最终设置：

> 将位置坐标损失中 w 和 h 分别取 平方根 来 代替 原本的 w 和 h。

![](https://img-blog.csdn.net/20180811073744141?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Rjcm1n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

上图可见，对于水平方向上同等尺度的增量，其x越小，其取根号之后产生的偏差越大。如图绿色段明显大于红色段。 换句话说，在小目标检测上，如果错了一点，损失函数会很大。而在大目标上，就算错了同样的量，损失函数也不会很大。

# 2. YOLOv1训练代码实现

下面我们通过代码来看Yolo如何实现网络的训练过程以及一些细节我们必须从代码来看这些细节。

## 2.1 数据下载

我们首先创建一个仓库：

![](https://flyman-cjb.oss-cn-hangzhou.aliyuncs.com/picgos/20190223141159.png)

然后切换到某个路径下:

```sh
git clone *仓库地址
cd Yolov1-Tensorflow
touch download.sh
```

在download.sh文件中添加如下内容：

```sh
#!/usr/bin/env bash
echo "Creating data directory..."
mkdir -p data && cd data
mkdir weights
mkdir pascal_voc
 
echo "Downloading Pascal VOC 2012 data..."
axel -n 30 http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
 
echo "Extracting VOC data..."
tar xf VOCtrainval_06-Nov-2007.tar
 
mv VOCdevkit pascal_voc/.
 
echo "Done."
```

我们需要先安装`axel`，`wget`下载太慢了，随后执行这个脚本，数据就会被下载：

```sh
sudo apt-get install axel
sh download.sh
```

此外我们希望一些文件，比如`.jpg`格式等文件不被git追踪，因为国内网络本来就慢，把这些添加进来一次push就要好久，另外Git本来也有100M大小限制。其主要通过`.gitignore`文件管理，Pycharm里面有个Git工具可以方便添加，主要有Python、Pycharm的一些临时文件，还有我们的数据以及预测结果。大家可以从我的github中下载。

## 2.2 主体代码

我们新建一个`yolo.py`文件，参数等为了方便大家学习就直接在文件中写了，不再用`config`文件管理。

**导入包**

```python
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
```

**基本代码**

我们建立了基本的代码，后面我们要在编写损失函数的时候再往上面添加新的变量以及方法，注意loss层我们没有写进去。我们主要完成了：

1. 图像、标签占位符
2. 推断过程（Net的主要结构）
3. leaky_relu

需要注意的有两点：

1. 是有关padding的知识，这里不展开叙述，可以查看我的笔记[<TensorFlow Padding知识>](http://db8f081b.wiz03.com/share/s/3rzMwr0yHAKp22o0Cy1XYjcD1dwJT01VyAFn2D-HEg0_cD09)。之所以刚开始会将图像从$448 \times 448 \times 3$ padding 到$454 \times 454\times 3$，以及后面padding个1，是为了Tensor的shape变化。
2. 由`[batch, image_height, image_width, channels]`变成`[batch, channels, image_height, image_width]`是为了Flatten的时候保持提取特征空间上的连续

跟YOLOv1原文的网络结构有些许差异，但不重要，我们主要学习它的思想以及Tensorflow编程：

```python
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
 
 
class YOLONET(object):
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.class_num = len(self.classes)
        self.image_size = 448
        self.cell_size = 7
        self.boxes_per_cell = 2
        self.leaky_alpha = 0.1  # 激活函数leaky_relu
        self.keep_prob = 0.5
        self.output_size = self.cell_size * self.cell_size * (5 * self.boxes_per_cell + self.class_num)
       # 1. 图像占位符
        self.images = tf.placeholder(dtype=tf.float32, name='images',
                                     shape=[None, self.image_size, self.image_size, 3])
        # 2. 推断过程(不包含损失函数)
        self.inference = self.build_network(images=self.images,
                                            output_size=self.output_size)
        if self.is_training:
            # 3. 标签占位符
            self.labels = tf.placeholder(tf.float32, name='labels',
                                         shape=[None, self.cell_size, self.cell_size, 5 + self.class_num])
            # 4. 损失函数层(还没有写)
            self.loss_layer(predicts=self.inference, labels=self.labels)
 
     def build_network(self,
                      images,
                      output_size,
                      scope='yolo_net'):
        with tf.variable_scope(scope):
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    activation_fn=YOLONET.leaky_relu(self.leaky_alpha),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
                net = tf.pad(images,
                             np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                             name='pad_1')
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(net,
                             np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(net,
                                   keep_prob=self.keep_prob,
                                   is_training=self.is_training,
                                   scope='dropout_35')
                net = slim.fully_connected(net,
                                           output_size,
                                           activation_fn=None,
                                           scope='fc_36')
        return net
 
    # TODO 损失函数
    def loss_layer(self, predicts, labels, scope='loss_layer'):
        pass
 
    @staticmethod
    def leaky_relu(alpha):
        def op(inputs):
            return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
        return op
```

我们搭建完了主体代码，下面的重点是损失函数的编写。

## 2.3 损失函数设计考量下 (2)

上面说过损失函数是怎么构成的，我们再回顾一下损失函数构成，并且补充一些细节：

1. 坐标损失
2. 置信度损失
3. 类别损失

![img](https://img-blog.csdn.net/20180606164516310)

我们在设计考量1中还知道:

1. **位置坐标损失** 和 **类别损失函数** 的权重比重为 **5 : 1**。
2. 在**不包含 object** 的cell计算损失时，**置信度损失**的**权重为0.5**，在**包含 object时，权重取1**。
3. 将位置坐标损失中 **w** 和 **h** 分别取 **平方根** 来 **代替** 原本的 w 和 h。

现在我们来补充一些细节:

### 2.3.1 类别损失

![](https://flyman-cjb.oss-cn-hangzhou.aliyuncs.com/picgos/20190225115800.png)

类别损失相对较为容易，每个cell中只有一个classification label，我们只有当这个cell中有label目标的时候，才会去计算这个损失。

### 2.3.2 中心坐标损失

![](https://flyman-cjb.oss-cn-hangzhou.aliyuncs.com/picgos/20190225110309.png)

该等式计算了相对于预测的边界框中心位置$x, y$的loss，$\lambda$暂且就看做一个给定的常数。(下面部分翻译过来有点变扭，就用原文来说)

> The function computes a sum over each bounding box predictor (*j = 0.. B*) of each grid cell (*i = 0 .. S^2*).

*𝟙 obj* is defined as follows:：

- 1，If an object is present in grid cell *i* and the *j* th bounding box predictor is “responsible” for that prediction
- 0, otherwise

本意是如果网格单元i中存在目标，则第j个边界框预测值对该预测有效。

也就是说，只要这个cell中在label中含有object，那么基于这个cell提取出的30维Tensor计算的坐标损失都是有效的。这里有两个bounding box。

但仔细思考一下，我们是否两个bounding box都是需要的？

> 最后预测的时候我们每个cell只需要1个bound box(详细看后面的预测代码编写的部分，因为多的也要抑制掉去)，因此我们训练的时候完全没有必要将另一个bounding box 添加到损失函数中。这样一来可以提高效率，二来可以使损失函数专注于优化更好的那个bounding box的坐标。

既然我们只需要1个bounding box，那我们如何去选择这个bounding box？继续看原文：

> YOLO predicts multiple bounding boxes per grid cell. At training time we only want one bounding box predictor to be responsible for each object. We assign one predictor to be “responsible” for predicting an object based on which prediction has the highest current IOU with the ground truth.

简单说在训练阶段，我们只需要计算出每个cell预测出来的2个Bounding box 和 label Box的IOU值，取最大的IOU的那个Bounding box进行损失函数计算即可。

### 2.3.3 置信度损失

我们之前提到过两点：

![](https://flyman-cjb.oss-cn-hangzhou.aliyuncs.com/picgos/20190225112550.png)

1. label的Confidence为0-1标签，代表某个cell是否含有object。Inference的时候，我们用IOU\*Pro作为预测Confidence。
2. 我们希望没有Object的cell对网络的贡献小一点，而有目标的cell对网络的贡献大一点。否则网络只要全部预测出没有Object损失就很小了。我们定的比例是1:2。

这里每个cell同样会预测出2个bounding box，也就是有两个Confidence。这里我们是不是也只需要一个？

不，主要有两个原因：

第一，我们后面预测的时候需要用到Confidence的信息，当Predict的时候，我们首先在每个cell中预测出2个bounding box,之后我们会计算 Confidence \* Classification 的值作为类别置信度，再根据这个类别置信度进行过滤。如果我们训练的时候Confidence的第二个值就不准，那么我们预测的时候也会一塌糊涂。

其次，在两个bounding box中回归Confidence都有利于网络更好地判断整个cell中是否有目标。

因此我们两个bounding box中的Confidence都需要。因此我们最终这部分损失函数，**当在一个单元格中有对象时，*𝟙 obj等*于1，否则取值为0。**

## 2.4 损失函数的输入输出 (3)

现在我们回过头来讲模型的输入，我们需要明确一下我们的输入的shape，也就是这两行代码：

```python
self.images = tf.placeholder(dtype=tf.float32, name='images',
                              shape=[None, self.image_size, self.image_size, 3])
self.inference = self.build_network(images=self.images,
                                    output_size=self.output_size)
self.labels = tf.placeholder(tf.float32, name='labels',
                          shape=[None, self.cell_size, self.cell_size, 5 + self.class_num])
```

- inference: `[Batch, 1470]`
- labels:` [Batch, 7, 7, 25]`

先暂时不要去管到底怎么从VOC格式的数据转换为Yolo格式，我们明确下我们的输入输出的含义。损失函数主要接受两个参数，主要是模型Inference出来的30维的Tensor以及labels的25维的Tensor，含义之前已经明确过。

### 2.4.1 Inference

主要30维度Tensor主要有以下点：

1. $x, y$ \* 2

   我们希望模型拟合的是以cell左上角为基准中心点所在cell的位置。比例基准是cell的大小。

2. $\sqrt{w}, \sqrt{h}$ \* 2

   我们希望模型拟合出的是object 宽高的根号值，这样好处是减小目标大小不一的影响。

3. Confidence \* 2

   Inference的置信度Confidence是由拟合出的Prob \* IOU 计算出来的。如果简单地拟合出Prob，即使预测地很偏也不会受到惩罚。

4. Classification 20

   整个cell所属的类别。

### 2.4.2 labels

标签存储的信息就简单很多，但是是基于全局坐标系的，主要包括：

1. 坐标(x, y, w, h )

   - x, y 为中心点相对于整张图片左上角的(归一化到0-1之间)
   - w,h 目标相对于整张图片的比例(归一化到0-1之间)

2. Confidence

   在标签中，它代表cell中是否存在目标，为0-1标签，只有目标中心落的那个网格标签为1，其余为0。

3. Classification

   为one-hot标签，代表这个cell归属的类别。如果这个cell里面没有目标，有些损失是不会被计算的。

**整个label的shape` [Batch, 7, 7, 5+20=25]`。预测出来的Tensor的Shape为`[Batch, 1470]`在损失函数层，我们会将其`reshape`为`[Batch, 7, 7, 2*(4+1)+20=30]`来方便整个损失函数的计算。**

### 2.4.3 labels的制作

先来制作labels文件吧，我们首先需要划分数据集(VOC已经帮我们划分好了)，labels的文件的输入是VOC格式，我们需要将其从VOC格式转换为Yolo格式，另外为了读取高效，我们会将其保存为`tfrecords`文件(不再仔细讲代码，我们默认你能够熟练使用Tensorflow)：

**1. VOC坐标转换**

```python
 
```

**2. records文件写入**

```python
 
```

 

## 2.5 损失函数的实现

2.3.2 损失函数层

我们先取出我们要的各个部分，在实现的时候label为None的时候存储的为0，后面我们会从零转换一下标签：

```python
    def loss_layer(self, predicts, labels, scope='loss_layer'):
        """
 
        Args:
            predicts: Net的输出 [Batch, 1470]
            labels: label[Batch, 7, 7, 25]
 
        """
        predicts = tf.reshape(predicts, name='predicts',
                              shape=[None, self.cell_size, self.cell_size, 5 * self.boxes_per_cell + self.class_num])
        # 1.0 预测的boxes x, y, w, h
        pre_boxes = predicts[..., : 4 * self.boxes_per_cell]
        # 1.1 预测的boxes confidence
        pre_confi = predicts[..., : 4 * self.boxes_per_cell : 5 * self.boxes_per_cell]
        # 1.2 预测的classification
        pre_class = predicts[..., 5 * self.boxes_per_cell:]
 
        # 2.0 label的x, y, w, h 其实只有一个
        lab_boxes = labels[..., :4]
        # 2.1 confidence标签, 同时也负责标记哪些cell中有目标,这些cell会被用来预测
        lab_confidence = labels[..., 4]
        # 2.2 类别标签
        lab_class = labels[..., 5:]
        ......
```

现在我们要构建损失函数了，我们从简单到复杂开始构建：

1. 类别损失

   我们提到类别损失，我们知道只有在有object的cell上，标签上才有类别标签，为一个20维的one-hot标签，其余的本应该为None。但是这样的结果是，整个数据需要存储为稀疏矩阵，而且计算机需要一一判断这里是否有值。为了容易计算，我们将其值存储为0。

   但是这又带来一个问题，当卷积网络在提取特征做预测的时候，这个Inference出的值并不是0，这样两个标签相减计算均方差就会给网络带来损失。事实上我们是不需要计算没有object网格的坐标损失的。

   因此我们将其乘以Confidence标签(也就是标记$7 \times 7​$ cell 中哪个cell有object，shape=`[batch, 7, 7, 1]`)，在没有目标的位置，这些计算出来的损失会被乘以0。虽然这样子效率会变低，但是这却是计算机矩阵化处理的技巧之一。

   让我们记住：

   > 我们只需要计算有object的cell中的类别损失，对于多余出来的计算，我们可以在矩阵相乘的时候乘以0来处理掉。

   关于Tensorflow的一些矩阵操作，不明白的可以查看我的笔记[TODO]，另外如果你对一些操作不太明确的话，建议你和我一样新建一个`tensor_experiment.py`文件，并且不将其加入版本控制，主要用来试验一些不确定的操作。

   下面我们看代码：

   ![](https://flyman-cjb.oss-cn-hangzhou.aliyuncs.com/picgos/20190224103334.png)

   ```python
   # 1. 类别损失
   # [batch, 7, 7, 1] * ([batch, 7, 7, 20] - [batch, 7, 7, 20]) 这里是逐元素相乘
   class_delta = lab_confidence * (pre_class - lab_class)
   # 除了Batch维度没有求sum,但之后仍旧平均了.
   class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                               name='class_loss') * 1
   ```

2. 置信度损失

   我们之前提到过三点：

   1. 每个cell不论它是否有目标，我们都会计算它的损失。每个bounding box 预测出来的confidence我们也都会利用起来。因为都有利于网络回归出某个cell中是否含有物体。
   2. label的confidence为0-1标签，代表某个cell是否含有object。预测的时候，我们用IOU\*Pro作为预测Confidence。
   3. 我们希望没有Object的cell对网络的贡献小一点，而有目标的cell对网络的贡献大一点。否则网络只要全部预测出没有Object损失就很小了。我们定的比例是1:2。

   ![](https://flyman-cjb.oss-cn-hangzhou.aliyuncs.com/picgos/20190224121408.png)

   既然我们要使用到IOU，那么我们需要将其先计算出来，不妨新建一个方法`calc_iou`吧，因为所有代码堆在一起，以后真的不大好维护：

   ```python
   # 后面我们补充
   def calc_iou(self, box1, box2, scope='iou'):
       pass
   ```

   > **IOU计算**

   IOU称为交并比（Intersection over Union），计算的是 “预测的边框” 和 “真实的边框” 的交集和并集的比值:

   ![img](https://img-blog.csdn.net/20180922220708895?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQwNjE2MzA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

   人类解决问题的方式都是从简单的入手的，之后再逐渐复杂，输入肯定是两个$x, y, w, h$，我们肯定需要将坐标系转换一下，变为标记左上角的$x_1,y_1$以及右下角的$x_2,y_2$，回忆下我们的坐标系，不同于一般坐标系，我们的**坐标系$y$轴方向向下**。

   首先的想法是：

   > 我们考虑两个边款的相对位置，然后按照相对位置分情况讨论来计算交集。

   ![img](https://img-blog.csdn.net/20180923221316210?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQwNjE2MzA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

   但是如果这样写程序的话，你确定你的导师不会打死你吗？我们来实现一版不让导师打死的IOU计算代码，我们将其矩阵化处理：

   我们先从X轴方向看，其实只存在3种情况：

   1. 两个box没有交集

      ![](https://flyman-cjb.oss-cn-hangzhou.aliyuncs.com/picgos/20190224131132.png)

   2. 部分重叠

      ![](https://flyman-cjb.oss-cn-hangzhou.aliyuncs.com/picgos/20190224192327.png)

   3. 完全重叠

      ![](https://flyman-cjb.oss-cn-hangzhou.aliyuncs.com/picgos/20190224192359.png)

   我们思考以下几个问题：

   - 我们如何计算交集的面积？

     > min($X_R$)-max($X_L$)即为交集的值，我们将其与0值比较，取最大值。(不需要判断，因为)

   - 我们如何计算并集的面积？

     > 其实非常简单，我们只需要求两个bounding box各自的面积，然后减掉交集即可。

   知道了这个思路，那么我们就可以着手开始实现了，在此之前，我们需要对坐标进行转换，从$x, y, w, h$转换为所需要的$x_1, y_1, x_2, y_2​$：

   ![](https://flyman-cjb.oss-cn-hangzhou.aliyuncs.com/picgos/20190224200504.png)

   注意我们计算这个IOU的时候，预测出的$x, y, w, h​$需要需要将其转换为全局坐标系下的$x_1, y_1, x_2, y_2​$，而不是以cell的左上角为基准的yolo坐标系。

   **事实上我们做label的时候,$x, y$是夹带着左上角基准点的信息的，也就是该点在原图中的位置比例，w以及h也是个基于全图的比例值。但我们希望特征图中每个cell预测的是基于切分后cell左上角的偏移量。**

    因此我们纠正一下之前的说法，主要是三点：

   > - **label的$x, y, w, h​$是基于全图的比例。**
   > - **inference出来的$x, y, w, h$是基于cell左上角的值，$x, y$是相对于cell大小的比例值，而$w, h​$是相对于整张图片的**
   > - **实际上模型inference出来的也并不是$w, h$，而是它们的根号值：$\sqrt{w}, \sqrt{h}$。我们需要将其平方回去**

   因为之后计算坐标损失的时候，我们也会需要用到这个坐标转化，对此我们新建一个方法`cell2global()`，先将inference出来的基于cell左上角的$x, y, \sqrt{w}, \sqrt{h} $坐标系转换为基于全图的$x, y, w, h$。

   ```python
   def cell2global(self, pre_boxes):
       pass
   ```

   既然程序的框架已经搭建好，我们来一步步实现吧：

   1. 我们首先将Inference出来的两个Bounding box坐标系进行转换:

      ```python
      def cell2global(self, pre_boxes):
              """
 
              Inputs:
                  模型Inference出来的两个bounding box 的x,y,sqrt(w), sqrt(h),shape=[batch, 7, 7, 8]
              Returns:
                  基于全局坐标系下的x, y, w, h
 
              """
              # 1. shape >>> [Batch, 7, 7, 2, 4]
              pre_boxes = tf.reshape(pre_boxes, shape=[None, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
              # 2. 沿着axis=1方向逐渐增大,我们希望shape=[1, 7, 7, 2, 1]
              offset = tf.tile(tf.range(self.cell_size))...  # 不会了...
      ```

       Note：写到这里的时候我不确定后面的操作是否是我想要的，因此我打开之前创建的实验文件，用tf验证一下我的实现是否是我想要的功能。一个很简单的操作是：

      ```python
      import tensorflow as tf
 
      offset = tf.range(7)
 
      with tf.Session() as sess:
          _offset = sess.run(offset)
          print(_offset)
          print(_offset.shape)
      >>> 输出
          [0 1 2 3 4 5 6]
          (7,)
      ```

      注意numpy跟Tensorflow广播规则还是有很大区别的，我们正式开启试验，在不断调试之后，最终达到我想要的要求(高维矩阵操作的时候，请不要尝试$x, y$二维平面来想象，我们直接抽象地进行运算)：

      ```python
      import tensorflow as tf
      # 我们希望shape=[1, 7, 7, 2, 1]
      # 1. 先变为 7 x 7,再以这个为基础再次进行复制
      offset_axis_2 = tf.tile(tf.expand_dims(tf.range(7), axis=0),
                              multiples=[7, 1])
      # 2. 7 x 7 >>> 1, 7, 7, 1, 1 >>> 1, 7, 7, 2, 1
      # 沿着axis=2的方向变大
      offset_axis_2 = tf.tile(tf.reshape(offset_axis_2, shape=[1, 7, 7, 1, 1]),
                              multiples=[1, 1, 1, 2, 1])
      # 沿着axis=1的方向变大
      offset_axis_1 = tf.transpose(offset_axis_2, (0, 2, 1, 3, 4))
 
      with tf.Session() as sess:
          _offset_axis_1 = sess.run(offset_axis_1)
          _offset_axis_2 = sess.run(offset_axis_2)
          print(_offset_axis_1.shape)
          print('offset_axis_1')
          print(_offset_axis_1[0, :, 0, 0, 0])
          print('offset_axis_2')
          print(_offset_axis_2[0, 0, :, 0, 0])
      >>> 输出
      (1, 7, 7, 2, 1)
      offset_axis_1
      [0 1 2 3 4 5 6]
      offset_axis_2
      [0 1 2 3 4 5 6]
      ```

      我们继续补充了这么一段函数：

      ```python
      def cell2global(self, pre_boxes):
              """
 
              Inputs:
                  模型Inference出来的两个bounding box 的x,y,sqrt(w), sqrt(h),shape=[batch, 7, 7, 8]
              Returns:
                  基于全局坐标系下的x, y, w, h
 
              """
              # 1. shape >>> [Batch, 7, 7, 2, 4]
              # TensorFlow中是以NHWC格式存储图片的,因此跟我们生成的矩阵是一致的
              pre_boxes = tf.reshape(pre_boxes, shape=[None, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
              # 2. 沿着axis=2的方向逐渐增大,我们希望shape=[1, 7, 7, 2, 1]
              offset_axis_2 = tf.tile(tf.expand_dims(tf.range(7), axis=0),
                                      multiples=[7, 1])
              offset_axis_2 = tf.tile(tf.reshape(offset_axis_2, shape=[1, 7, 7, 1, 1]),
                                      multiples=[1, 1, 1, 2, 1])
              # 3. 沿着axis=1的方向变大
              offset_axis_1 = tf.transpose(offset_axis_2, (0, 2, 1, 3, 4))
              global_boxes = tf.stack([(pre_boxes[..., 0] + ???) / self.cell_size])
      ```

   2. 我们再利用这个值去加回到labels的$x, y$的时候，我们发现在选择$x$加上某个偏移量的时候有了困难，主要是：

      我们不知道经过之前网络的`tf.flatten`以及后来我们自己将`[Batch, 1470]`先`reshape`为`[Batch, 7, 7, 30]`并从中取出`[Batch, 7, 7, 8]`并且最终`reshape`为`[Batch, 7, 7, 2, 4]`之后，这两个$7, 7$和原图的关系。

      其实不用担心，因为网络经过全连接层后，所有的图片和特征图对应的关系都乱了。卷积层之前有每个特征图的cell有感受野的概念，但对于全连接层来说，每个units都连接着前面所有的units，因此Reshape之后的Tensor主要看你想要拟合什么，关键是Test的时候统一即可。我们按照使用图像存储时候的`HW`的格式制作的：

      ```python
      global_boxes = tf.stack([(pre_boxes[..., 0] + offset_axis_2) / self.cell_size,
                                       (pre_boxes[..., 1] + offset_axis_1) / self.cell_size,
                                       tf.square(pre_boxes[..., 2]),
                                       tf.square(pre_boxes[..., 3])], axis=-1)
      ```

      Flatten之后的信息全部乱了，之后还要Reshape回来，这是一个YOLOv1的缺点之一。

 

 

 
