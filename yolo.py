import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from configs import CLASS


class YOLONET(object):
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.classes = CLASS
        self.class_num = len(self.classes)
        self.image_size = 448
        self.cell_size = 7
        self.boxes_per_cell = 2
        self.leaky_alpha = 0.1  # 激活函数leaky_relu
        self.keep_prob = 0.5
        self.output_size = self.cell_size * self.cell_size * (5 * self.boxes_per_cell + self.class_num)
        # 权重系数
        self.class_scale = 1
        self.object_confidence_scale = 1
        self.no_object_confidence_scale = 0.5
        self.coord_scale = 5

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
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total loss', self.total_loss)


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
                net = slim.dropout(net, keep_prob=self.keep_prob,
                                        is_training=self.is_training,
                                   scope='dropout_35')
                net = slim.fully_connected(net,
                                           output_size,
                                           activation_fn=None,
                                           scope='fc_36')
                # [Batch, 7, 7, 30] reshape 预测结果
                net = tf.reshape(net, name='predicts', shape=[None, self.cell_size, self.cell_size,
                                                                   5 * self.boxes_per_cell + self.class_num])
        return net

    def loss_layer(self, predicts, labels, scope='loss'):
        """

        Args:
            predicts: 网络的输出 [Batch, 7, 7, 30]
            labels: label[Batch, 7, 7, 25], [Batch, (h方向), (w方向), 25]
                    为与计算机存储图片格式相同,我们存储标签的时候先按 h / w 顺序存储.
        """
        with tf.name_scope('Predict Tensor'):
            ############## 预测 ##############
            """
            1. 预测坐标: x, y 基于cell, sqrt(w), sqrt(h) 基于全图 (0-1)范围内
            2. 2个bounding box,拥有2个坐标以及置信度
            """
            # 1. Bounding box 坐标预测 [batch, 7, 7, :8] >> shape=[batch, 7, 7, 2, 4]
            pre_boxes = tf.reshape(predicts[..., : 4 * self.boxes_per_cell],
                                   shape=[None, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
            # 2. Bounding box 置信度预测 [batch, 7, 7, 8:10] >> shape=[batch, 7, 7, 2]
            pre_confidence = predicts[..., 4 * self.boxes_per_cell: 5 * self.boxes_per_cell]
            # 3. class 类别预测 [batch, 7, 7, 10:] >> shape=[batch, 7, 7, 20]
            pre_class = predicts[..., 5 * self.boxes_per_cell:]
        with tf.name_scope('Label Tensor'):
            ############## 标签 ##############
            """
            1. 标签坐标: x,y,w,h 均基于全图(0-1)
            """
            # 1. box response_label [batch, 7, 7, 0]  >> shape=[batch,7, 7, 1]
            # lab_response 只负责标记cell中是否有object,置信度标签需要跟IOU实时计算
            lab_response = labels[..., 0]
            # 2. box 坐标label [batch, 7, 7, 1:5] >> shape=[batch, 7, 7, 2, 4]
            lab_boxes = tf.reshape(labels[..., 1:5],
                                   shape=[None, self.cell_size, self.cell_size, 1, 4])
            lab_boxes = tf.tile(lab_boxes, [1, 1, 1, self.boxes_per_cell, 1])
            # 3. class 类别标签 [batch, 7, 7, 5:] >> shape=[batch, 7, 7, 20]
            lab_class = labels[..., 5:]

        ############## 损失函数 ##############
        with tf.variable_scope(scope):
            # 1. 类别损失
            class_loss = self.class_loss(pre_class, lab_class, lab_response)
            # 2. 坐标转换基于cell的x, y, sqrt(w), sqrt(h) >> 基于全图的x, y, w, h
            global_pre_boxes = self.pre_to_label_coord(pre_boxes)  # [batch, 7, 7, 2, 4]
            # 3. 计算iou shape=[batch, 7, 7, 2]
            iou_pre_label = self.calc_iou(global_pre_boxes, lab_boxes)
            # 4. 目标掩模 / 非目标掩模
            object_mask, no_object_mask = self.mask(iou_pre_label, lab_response)
            # 5. 置信度损失
            object_confidence_loss, no_object_confidence_loss = self.confidence_loss(
                pre_confidence, iou_pre_label, object_mask, no_object_mask)
            # 6. 坐标损失
            coord_loss = self.coord_loss(pre_boxes, lab_boxes, object_mask)
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_confidence_loss)
            tf.losses.add_loss(no_object_confidence_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_confidence_loss', object_confidence_loss)
            tf.summary.scalar('no_object_confidence_loss', no_object_confidence_loss)
            tf.summary.scalar('coord_loss', coord_loss)

    def class_loss(self, pre_class, lab_class, lab_response):
        """

        Args:
            pre_class: 预测类别 [batch, 7, 7, 20]
            lab_class: 标签类别 [batch, 7, 7, 20] 20维 one-hot标签
            lab_response: 标记某个Cell中是否有object [batch, 7, 7, 1]
            weight: 权重系数之前说过,类别损失和坐标损失的比重是1 : 5

        Returns:
            类别损失
        """
        # 乘以 label_response 来去除掉没有Object位置的类别损失
        # 注意一个cell中是否有物体是通过置信度损失来回归的
        # 这种分开的思想让不同位置的参数回归不同属性,而不是把它们融合在一起
        with tf.name_scope('class loss'):
            delta = lab_response * (pre_class - lab_class)
            class_loss = self.class_scale * tf.reduce_mean(tf.reduce_sum(tf.square(delta), axis=[1, 2, 3]),
                                                           name='class_loss')
        return class_loss

    def confidence_loss(self,
                        pre_confidence,
                        iou_pre_label,
                        object_mask,
                        no_object_mask,
                        ):
        """

        Args:
            pre_confidence: 预测置信度 shape=[batch, 7, 7, 2]
            iou_pre_label: IOU shape=[batch, 7, 7, 2]
            object_mask: 目标掩模 [batch, 7, 7, 2] 有目标的位置是1,其余为0
            no_object_mask: 非目标掩模 [batch, 7, 7, 2] 没有目标的位置是1,其余为0
        """
        with tf.name_scope('Confidence loss'):
            with tf.name_scope("Object Confidence loss"):
                # 用目标掩模进行判断是否有目标
                object_confidence_delta = object_mask * (pre_confidence - iou_pre_label)
                object_confidence_loss = self.object_confidence_scale * tf.reduce_mean(
                    tf.reduce_sum(tf.square(object_confidence_delta), axis=[1, 2, 3]))
            with tf.name_scope('No Object Confidence loss'):
                # 只要预测出置信度就是错的,我们用掩模抑制
                no_object_confidence_delta = no_object_mask * pre_confidence
                no_object_confidence_loss = self.no_object_confidence_scale * tf.reduce_mean(
                    tf.reduce_sum(tf.square(no_object_confidence_delta), axis=[1, 2, 3]))
        return object_confidence_loss, no_object_confidence_loss

    def coord_loss(self,
                   pre_boxes,
                   lab_boxes,
                   object_mask):
        """

        Args:
            pre_boxes: [batch, 7, 7, 2, 4] 基于cell的x, y以及全图 sqrt(w), sqrt(h)
            lab_boxes: [batch, 7, 7, 2, 4] 基于全图的x, y, w, h
            object_mask: [batch, 7, 7, 2]

        Returns:

        """
        with tf.name_scope('Coord loss'):
            coord_mask = tf.expand_dims(object_mask, axis=-1)
            cell_lab_boxes = self.label_to_pre_cood(lab_boxes)
            coord_delta = coord_mask * (pre_boxes - cell_lab_boxes)
            coord_loss = self.coord_scale * tf.reduce_mean(
                tf.reduce_sum(tf.square(coord_delta), axis=[1, 2, 3, 4]))
        return coord_loss

    def mask(self, iou_pre_label, label_response):
        """

        Args:
            iou_pre_label: 两个 Bounding box 的 IOU [batch, 7, 7, 2]
            label_response: [batch, 7, 7, 1]  0-1

        Returns:
            object_mask: 有目标的掩模,有object并且IOU最高的bounding box [batch, 7, 7, 2]
            no_object_mask: 其余情况

        """
        object_mask = tf.reduce_max(iou_pre_label, axis=-1, keep_dims=True)
        object_mask = tf.cast((iou_pre_label>= object_mask), tf.float32)
        object_mask = object_mask * label_response  # 还需要乘以 response
        no_object_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
        return object_mask, no_object_mask

    def pre_to_label_coord(self, pre_boxes):
        """坐标转换基于cell的x, y, sqrt(w), sqrt(h) >> 基于全图的x, y, w, h"""

        # 1. 沿着axis=2的方向逐渐增大,我们希望shape=[1, 7, 7, 2(bounding box), 1]1
        offset_axis_2 = tf.tile(tf.expand_dims(tf.range(7), axis=0),
                                multiples=[7, 1])
        offset_axis_2 = tf.tile(tf.reshape(offset_axis_2, shape=[1, 7, 7, 1, 1]),
                                multiples=[1, 1, 1, 2, 1])
        # 3. 沿着axis=1的方向变大
        offset_axis_1 = tf.transpose(offset_axis_2, (0, 2, 1, 3, 4))

        w = tf.square(pre_boxes[..., 2])
        h = tf.square(pre_boxes[..., 3])

        # 4. 计算x的时候.因为图像是以h, w格式存储的,也就是 x变化 在axis=2上递增
        global_x = (pre_boxes[..., 0] + offset_axis_2) / self.cell_size
        global_y = (pre_boxes[..., 1] + offset_axis_1) / self.cell_size
        global_pre_boxes = tf.stack([global_x, global_y, w, h], axis=-1)
        return global_pre_boxes

    def label_to_pre_cood(self, lab_boxes):
        offset_axis_2 = tf.tile(tf.expand_dims(tf.range(7), axis=0),
                                multiples=[7, 1])
        offset_axis_2 = tf.tile(tf.reshape(offset_axis_2, shape=[1, 7, 7, 1, 1]),
                                multiples=[1, 1, 1, 2, 1])
        offset_axis_1 = tf.transpose(offset_axis_2, (0, 2, 1, 3, 4))
        sqrt_w = tf.sqrt(lab_boxes[..., 2])
        sqrt_h = tf.sqrt(lab_boxes[..., 3])
        cell_x = lab_boxes[..., 0] * self.cell_size - offset_axis_2
        cell_y = lab_boxes[..., 1] * self.cell_size - offset_axis_1
        cell_lab_boxes = tf.stack([cell_x, cell_y, sqrt_w, sqrt_h], axis=-1)
        return cell_lab_boxes

    def calc_iou(self, boxes1, boxes2, scope='IOU'):
        """ 计算训练时候 bounding box 和 label 的IOU

        Args:
            boxes1: 预测 Boxes
                    [Batch, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] / [x, y, w, h]
            boxes2: label Boxes
                    [Batch, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] /[x, y, w, h]
            scope: 我们

        Returns:
             IOU: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            """transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            1. 涉及矩阵操作的时候,Tesnorflow一般只是将前面的维度当做Batch
            2. 不涉及矩阵的操作,我们可以拿出单个元素考虑,之后将其矩阵化
            """
            boxes1_voc = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                   boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                   boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                   boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                   axis=-1)
            boxes2_voc = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                   boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                   boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                   boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                   axis=-1)

            # calculate the left up point & right down point
            # [batch, 7, 7, 2, 2] 对应位置最大值, maximum 支持广播,但不能指定轴
            lu = tf.maximum(boxes1_voc[..., :2], boxes2_voc[..., :2])  # x1, y1 max(X_L)
            rd = tf.minimum(boxes1_voc[..., 2:], boxes2_voc[..., 2:])  # x2, y2 min(X_R)

            # [batch, 7, 7, 2, 2]   min(X_R)-max(X_L)
            intersection = tf.maximum(0.0, rd - lu)
            # [batch, 7, 7, 2] 2个bounding box跟label的IOU,因此这里有2个
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square by w*h
            # [batch, 7, 7, 2] * [batch, 7, 7, 2]
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]
            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    @staticmethod
    def leaky_relu(alpha):
        def op(inputs):
            return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
        return op
