#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

directory_root = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_path = os.path.join(directory_root, "data")
model_path = os.path.join(directory_root, "models")
image_path = os.path.join(directory_root, "images")

class Config():

    model_dir = model_path
    image_dir = image_path
    inception_v3_image_size = 299
    inception_v1_image_size = 299
