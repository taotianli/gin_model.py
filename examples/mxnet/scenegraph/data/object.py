"""Pascal VOC object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import warnings
import json
import pickle
import numpy as np
import mxnet as mx
from gluoncv.data import COCODetection
from collections import Counter

class VGObject(COCODetection):
    CLASSES = ["airplane", "animal", "arm", "bag", "banana", "basket", "beach",
               "bear", "bed", "bench", "bike", "bird", "board", "boat", "book",
               "boot", "bottle", "bowl", "box", "boy", "branch", "building", "bus",
               "cabinet", "cap", "car", "cat", "chair", "child", "clock", "coat",
               "counter", "cow", "cup", "curtain", "desk", "dog", "door", "drawer",
               "ear", "elephant", "engine", "eye", "face", "fence", "finger", "flag",
               "flower", "food", "fork", "fruit", "giraffe", "girl", "glass", "glove",
               "guy", "hair", "hand", "handle", "hat", "head", "helmet", "hill",
               "horse", "house", "jacket", "jean", "kid", "kite", "lady", "lamp",
               "laptop", "leaf", "leg", "letter", "light", "logo", "man", "men",
               "motorcycle", "mountain", "mouth", "neck", "nose", "number", "orange",
               "pant", "paper", "paw", "people", "person", "phone", "pillow", "pizza",
               "plane", "plant", "plate", "player", "pole", "post", "pot", "racket",
               "railing", "rock", "roof", "room", "screen", "seat", "sheep", "shelf",
               "shirt", "shoe", "short", "sidewalk", "sign", "sink", "skateboard",
               "ski", "skier", "sneaker", "snow", "sock", "stand", "street",
               "surfboard", "table", "tail", "tie", "tile", "tire", "toilet",
               "towel", "tower", "track", "train", "tree", "truck", "trunk",
               "umbrella", "vase", "vegetable", "vehicle", "wave", "wheel",
               "window", "windshield", "wing", "wire", "woman", "zebra"]
    def __init__(self, **kwargs):
        super(VGObject, self).__init__(**kwargs)
    
    @property
    def annotation_dir(self):
        return ''

    def _parse_image_path(self, entry):
        dirname = 'VG_100K'
        filename = entry['file_name']
        abs_path = os.path.join(self._root, dirname, filename)
        return abs_path
