# coding: utf-8
# !/usr/bin/env python

import imageio
import numpy as np
import os


def generate_gif(filenames, gifname="test_simulation", ext="png"):
    images = []
    for filename in filenames:
        images.append(imageio.imread("img/%s.%s" % (filename, ext)))
    imageio.mimsave('anim/%s.gif' % gifname, images)

    for filename in filenames:
        os.remove("img/%s.%s" % (filename, ext))
