import os
import numpy as np
from .utils import load_data
from .utils.read import DataSet
from .utils.sample import Sample
from .utils.tools import generateN2N, load_triple_original_file

from .utils.test import calculate_ranks_on_valid_via_triple
from .utils.test import calculate_ranks_on_valid_via_pair
from .utils.test import calculate_ranks_on_test_via_triple
from .utils.test import calculate_ranks_on_test_via_pair

from typing import List, Tuple, Union, Any, Optional