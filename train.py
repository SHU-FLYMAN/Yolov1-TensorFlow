import os
import argparse
import tensorflow as tf
import configs as cfg
from yolo import YOLONET
from utils import To_tfrecords, Dataset, ShowImageLabel


def train():
    to_tfrecord = To_tfrecords(txt_file='trainval.txt')
    to_tfrecord.transform()
    train_generator = Dataset(filenames='data/tfr_voc/trainval.tfrecords',
                              enhance=True)
    train_dataset = train_generator.transform()
    yolo = YOLONET()


if __name__ == '__main__':
    train()
