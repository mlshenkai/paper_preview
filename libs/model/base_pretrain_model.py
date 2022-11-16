# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/10 9:10 PM
# @File: base_pretrain_model
# @Email: mlshenkai@163.com

import gc
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from packaging import version
from torch import Tensor, device, nn
from torch.nn import CrossEntropyLoss
from transformers.generation_utils import GenerationMixin
from transformers.utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
from transformers.utils.import_utils import is_sagemaker_mp_enabled


