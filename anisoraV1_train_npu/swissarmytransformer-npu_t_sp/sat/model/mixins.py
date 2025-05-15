# -*- encoding: utf-8 -*-
'''
@File    :   mixins.py
@Time    :   2021/10/01 17:52:40
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import torch
from .base_model import BaseMixin
from .cached_autoregressive_model import CachedAutoregressiveMixin
from .finetune import *
from .attention import *