#!usr/bin/env/python
"""
Authors: Fatih Uenal & Shashi Badloe
Description: Launch tensorboard from stored model
Source of the model: https://link.springer.com/article/10.1007%2Fs10113-020-01677-8
"""
import tensorflow as tf
from tensorboard import main as tb
import os

def launch_tensorboard(path):
    """Launches tensorboard on localhost port 6006"""
    path = path +'/'+ os.listdir(path)[0]
    print("Loading model from '{}'".format(path))
    tf.flags.FLAGS.logdir = path
    tb.main()
    
if __name__ == '__main__':
    model_directory = '../tensorflow/logdir'
    launch_tensorboard(model_directory)
