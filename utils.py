import os
import tensorflow as tf
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from tqdm import tqdm



class To_tfrecords(object):
    def __init__(self, usage='train',
                 load_folder='data/pascal_voc/VOCdevkit/VOC2007',
                 save_folder='data/tfr_voc'):
        self.load_folder = load_folder
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.usage = usage
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.class_to_index = {_class: _index for _index, _class in enumerate(self.classes)}

    def transform(self):
        # 1. 获取作为训练集/验证集的图片编号
        if self.usage == 'train':
            txt_file = os.path.join(self.load_folder,
                                    'ImageSets',
                                    'Main',
                                    'trainval.txt')
        else:
            raise ValueError("We only support transform train step now")
        with open(txt_file) as f:
            image_index = [_index.strip() for _index in f.readlines()]

        # 2. 开始循环写入每一张图片以及标签到tfrecord文件
        with tf.python_io.TFRecordWriter(os.path.join(self.save_folder, self.usage + '.tfrecords')) as writer:
            for _index in tqdm(image_index, desc='开始写入tfrecords数据'):
                filename = os.path.join(self.load_folder, 'JPEGImages', _index) + '.jpg'
                xml_file = os.path.join(self.load_folder, 'Annotations', _index) + '.xml'
                img = tf.gfile.FastGFile(filename, 'rb').read()
                # 解析label文件
                label = self._parser_xml(xml_file)
                filename = filename.encode()
                # 需要将其转换一下用str >>> bytes encode()
                label = [float(_) for _ in label]
                # Example协议
                example = tf.train.Example(
                    features=tf.train.Features(feature={
                    'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
                    'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                    'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
                    }))
                writer.write(example.SerializeToString())

    def _parser_xml(self, xml_file):
        tree = ET.parse(xml_file)
        # 得到某个xml_file文件中所有的object
        objs = tree.findall('object')
        label = []
        for obj in objs:
            """ 
            <object>
                <name>chair</name>
                <pose>Rear</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>263</xmin>
                    <ymin>211</ymin>
                    <xmax>324</xmax>
                    <ymax>339</ymax>
                </bndbox>
            </object>
            """
            category = obj.find('name').text.lower().strip()
            class_id = self.class_to_index[category]

            bndbox = obj.find('bndbox')
            """
            <bndbox>
                <xmin>263</xmin>
                <ymin>211</ymin>
                <xmax>324</xmax>
                <ymax>339</ymax>
            </bndbox>
            """
            x1 = bndbox.find('xmin').text
            y1 = bndbox.find('ymin').text
            x2 = bndbox.find('xmax').text
            y2 = bndbox.find('ymax').text
            label.extend([x1, x2, y1, y2, class_id])
        return label


class Dataset(object):
    def __init__(self, filenames,
                 batch_size=32,
                 usage='train',
                 enhance=False):
        self.filenames = filenames
        self.batch_size = batch_size
        self.usage = usage
        self.enhance = enhance
        self.image_size = 448
        self.cell_size = 7

    def read(self):
        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(Dataset._parser)
        # TODO 数据增强
        # 2. 数据对图片以及标签进行处理
        dataset = dataset.map(lambda image, label:
                              tuple(tf.py_func(func=self._process,
                                               inp=[image, label],
                                               Tout=[tf.uint8, tf.float32])))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        return dataset

    # 对图像进行处理
    def _process(self, image, label):
        label = np.reshape(label, (-1, 5))
        label = [list(label[row, :]) for row in range(label.shape[0])]
        bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(*_) for _ in label], shape=image.shape)

        # 1. 数据增强(一些shape或者bounding box会发生变化的,否则用tensorflow自带的API)
        image, bbs = self._enhance(image, bbs)
        # 2. 图像resize
        image, bbs = self._resize(image, bbs)
        # 3. 制作yolo标签
        label = self._to_yolo(bbs)
        return image, label

    def _to_yolo(self, bbs):
        """

        Args:
            bbs:#标记类别，pascal_voc数据集一共有20个类，哪个类是哪个，则在响应的位置上的index是1

        Returns: [7, 7, 25]

        """
        label = np.zeros(shape=(self.cell_size, self.cell_size, 25), dtype=np.float32)

        for bounding_box in bbs.bounding_boxes:
            x_center = bounding_box.center_x
            y_center = bounding_box.center_y
            h = bounding_box.height
            w = bounding_box.width
            class_id = bounding_box.label
            x_ind = int((x_center / self.image_size) * self.cell_size)
            y_ind = int((y_center / self.image_size) * self.cell_size)
            # 对每个object,如果这个cell中有object了,则跳过标记
            if label[y_ind, x_ind, 0] == 1:
                continue
            # 1. confidence标签(对每个object在对应位置标记为1)
            label[y_ind, x_ind, 0] = 1
            # 2. 设置标记的框，框的形式为(x_center, y_center, width, height)

            label[y_ind, x_ind, 1:5] = [x_center, y_center, w, h]
            # 3. 标记类别，pascal_voc数据集一共有20个类，哪个类是哪个，则在响应的位置上的index是1
            label[y_ind, x_ind, int(5 + class_id)] = 1
        return label

    def _resize(self, image, bbs):
        image = ia.imresize_single_image(image, sizes=(self.image_size, self.image_size))
        bbs = bbs.on(image)
        return image, bbs

    @staticmethod
    def _enhance(image, bbs):
        """主要是一些图像增强之后Bounding box会发生变化的"""
        seq = iaa.Sequential([
            iaa.Crop(percent=(0, 0.1)),
            iaa.Flipud(0.3),
            iaa.Fliplr(0.3)])
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_image(image)
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
        return image_aug, bbs_aug

    @staticmethod
    def _parser(record):
        features = {"img": tf.FixedLenFeature((), tf.string, default_value=""),
                    "label": tf.VarLenFeature(tf.float32)}
        features = tf.parse_single_example(record, features)
        img = tf.image.decode_jpeg(features["img"])
        label = features["label"].values
        return img, label


def draw_box(image, bbs):
    """ 绘制图片以及对应的bounding box

    Args:
        img: numpy array
        boxes: BoundingBoxesOnImage对象

    """
    image_bbs = bbs.draw_on_image(image)
    print(bbs.bounding_boxes)
    plt.imshow(image_bbs)
    plt.show()


if __name__ == '__main__':
    # to_tfrecord = To_tfrecords(usage='train')
    # to_tfrecord.transform()
    _dataset = Dataset(filenames='data/tfr_voc/train.tfrecords')
    data = _dataset.read()
    iterator = data.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        for i in range(100):
            _label = sess.run(next_element)
            print(_label[0].shape)
            print(_label[1].shape)


