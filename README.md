# 飞翔的荷兰人带你入门目标检测第一季

> 论文地址: https://arxiv.org/pdf/1506.02640.pdf
>
> 笔记地址: https://github.com/SHU-FLYMAN/Yolov1-TensorFlow/docs/yolov1.html

这个系列，暂且叫做《**飞翔的荷兰人带你入门目标检测**：第一季》，是为了帮助更多的人入门目标检测的，第一季我们将讲完整个YOLO系列。

我们假设：

1. 你有一定的Tensorflow基础
2. 你有一定的Git的使用经验
3. 你对Deep learning有一定深入的了解(必备)

除了第三条，剩下的都可以一边学习一边补充。因为这个教程也是我自己复现Yolov1的学习笔记。此前一直使用Keras做网络结构方面的研究，Git也没什么经验，开发过机器学习系统，效果还不错，部署到了阿里云上，拿过一个竞赛的第10名，但水平说实话吧确实堪忧。

------

说到 Yolov3，必须从v1说起，很多文章都没有讲清楚细节。想要真正弄明白就要从v1说起，因为很多实现细节等等论文v3中都没有提到过，更多的细节需要自己手敲一遍代码才能明白。

速成的事情很少，但凡事都有例外，我们这里就有个速成的Yolov3教程，包括代码跟实现。

第一季的第一期我们会讲的啰嗦一点，务必让零基础的同学能够跟着入门。当讲到后面几期的时候，我们会加快讲速度，也会给大家补充网络的训练技巧，包括如何使用预训练权重、如何分布式训练模型等等技巧，不过这要等到这一季的第二期了，静等TensorFlow 2.0出来，2.0做了很大的改变，不再那么反人类了。

Yolov1 教程只是简单地训练网络，而且我们没有细讲怎么训练网络，因此效果并不会特别好，但是简单的网络更容易带我们入门Deep Learning，而且很多思想都一脉相承，理解Yolov1 有助于我们理解 Yolov3。

## Usage

我们不建议你使用`pip`来管理包,虽然也有`requirements.txt`,你可以通过`pip install -r requirements.txt`来安装所有依赖包.

我们更建议你用conda新建一个环境来管理你所需要的包:

```sh
git clone https://github.com/SHU-FLYMAN/Yolov1-TensorFlow.git
conda creat -n flyman python=3.6  # 创建python3.6虚拟环境
conda activate flyman  # 激活环境
cd Yolov1-TensorFlow
# 你可以选择这样安装
conda install --yes --file requirements.txt
# 或者 Conda会自动解决你所需要的依赖项
conda install 你需要的包
```

推荐你从零构建,也就是从`utils.py`对照着笔记来实现.

------

## TODO

我们除了学会Yolo的实现过程外，还包括学习到:

- [x] **Tfrecords** 数据集制作
- [x] **xml** 文件解析
- [x] **imgaug** 数据增强
- [x] **dataset** API 基本使用
- [x] 坐标转换
- [x] IOU 计算
- [x] Mask 掩模计算

------

我们还没有实现的:

- [ ] 网络训练技巧
- [ ] 分布式训练技巧
- [ ] 高效实现读取数据(没有使用`tf.Estimator`)
- [ ] 极大值抑制
- [ ] 多尺度
- [ ] TensorFlow Debug

其实学习最好的方式就是看别人的源代码然后自己做一些笔记, TensorFlow官网也是一个很值得学习的地方。后面我们将会实现Yolov3, 在这个教程中,我们将会用从零训练一个Yolov3网络,并且准确率最终会达到start-of-art.

