from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch


class Logger(object):
    def __init__(self, log_path):
        self.log = open(log_path, 'w')

    def write(self, txt):
        self.log.write(txt)
        # self.log.write('\n')
        self.log.flush()
        # print(txt)

    def close(self):
        self.log.close()

