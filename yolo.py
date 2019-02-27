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
                # NhWC变为NCHW,以矩阵最后个axis为方向展开
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




    # # TODO 损失函数
    # def loss_layer(self, predicts, labels, scope='loss'):
    #     """
    #
    #     Args:
    #         predicts: Net的输出 [Batch, 1470]
    #         labels: label[Batch, 7, 7, 25]
    #
    #     """
    #     predicts = tf.reshape(predicts, name='predicts',
    #                           shape=[None, self.cell_size, self.cell_size, 5 * self.boxes_per_cell + self.class_num])
    #     # 1.0 预测的boxes 2个 x, y, w, h
    #     pre_boxes = predicts[..., : 4 * self.boxes_per_cell]
    #     # 1.1 预测的boxes confidence
    #     pre_confi = predicts[..., : 4 * self.boxes_per_cell : 5 * self.boxes_per_cell]
    #     # 1.2 预测的classification
    #     pre_class = predicts[..., 5 * self.boxes_per_cell:]
    #
    #     # 2.0 label的x, y, w, h 其实只有一个
    #     lab_boxes = labels[..., :4]
    #     # 2.1 confidence标签, 同时也负责标记哪些cell中有目标,这些cell会被用来预测
    #     lab_confidence = labels[..., 4]
    #     # 2.2 类别标签
    #     lab_class = labels[..., 5:]
    #
    #     with tf.variable_scope(scope):
    #         # 1. 类别损失
    #         # [batch, 7, 7, 1] * ([batch, 7, 7, 20] - [batch, 7, 7, 20]) 这里是逐元素相乘
    #         class_delta = lab_confidence * (pre_class - lab_class)
    #         # 除了Batch维度没有求sum,但之后仍旧平均了.
    #         class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
    #                                     name='class_loss') * 1
    #         # 2. 置信度损失
    #         # 2.1 我们首先需要讲坐标系进行转换
    #         global_boxes = self.cell2global(pre_boxes)
    #
    #
    # def calc_iou(self, boxes1, boxes2, scope='iou'):
    #     with tf.variable_scope(scope):
    #         # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
    #         boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
    #                              boxes1[..., 1] - boxes1[..., 3] / 2.0,
    #                              boxes1[..., 0] + boxes1[..., 2] / 2.0,
    #                              boxes1[..., 1] + boxes1[..., 3] / 2.0],
    #                             axis=-1)
    #
    #         boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
    #                              boxes2[..., 1] - boxes2[..., 3] / 2.0,
    #                              boxes2[..., 0] + boxes2[..., 2] / 2.0,
    #                              boxes2[..., 1] + boxes2[..., 3] / 2.0],
    #                             axis=-1)
    #
    #         # calculate the left up point & right down point
    #         lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
    #         rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])
    #
    #         # intersection
    #         intersection = tf.maximum(0.0, rd - lu)
    #         inter_square = intersection[..., 0] * intersection[..., 1]
    #
    #         # calculate the boxs1 square and boxs2 square
    #         square1 = boxes1[..., 2] * boxes1[..., 3]
    #         square2 = boxes2[..., 2] * boxes2[..., 3]
    #
    #         union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
    #
    #     return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
    #
    # def cell2global(self, pre_boxes):
    #     """
    #
    #     Inputs:
    #         模型Inference出来的两个bounding box 的x,y,sqrt(w), sqrt(h),shape=[batch, 7, 7, 8]
    #     Returns:
    #         基于全局坐标系下的x, y, w, h
    #
    #     """
    #     # 1. shape >>> [Batch, 7, 7, 2, 4]
    #     # TensorFlow中是以NHWC格式存储图片的,因此跟我们生成的矩阵是一致的
    #     pre_boxes = tf.reshape(pre_boxes, shape=[None, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
    #     # 2. 沿着axis=2的方向逐渐增大,我们希望shape=[1, 7, 7, 2, 1]
    #     offset_axis_2 = tf.tile(tf.expand_dims(tf.range(7), axis=0),
    #                             multiples=[7, 1])
    #     offset_axis_2 = tf.tile(tf.reshape(offset_axis_2, shape=[1, 7, 7, 1, 1]),
    #                             multiples=[1, 1, 1, 2, 1])
    #     # 3. 沿着axis=1的方向变大
    #     offset_axis_1 = tf.transpose(offset_axis_2, (0, 2, 1, 3, 4))
    #     # [batch, 7, 7, 2, 1] x  有 7 x 7个cell,每个cell中都可能有对象
    #     # 这里
    #     global_boxes = tf.stack([(pre_boxes[..., 0] + offset_axis_2) / self.cell_size,
    #                              (pre_boxes[..., 1] + offset_axis_1) / self.cell_size,
    #                              tf.square(pre_boxes[..., 2]),
    #                              tf.square(pre_boxes[..., 3])], axis=-1)
    #     return global_boxes











    @staticmethod
    def leaky_relu(alpha):
        def op(inputs):
            return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')

        return op
