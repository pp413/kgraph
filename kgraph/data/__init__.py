#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/11/26 17:51:49
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#

# utils
from .data_utils import get_download_dir, set_download_dir
from .data_utils import load_from_text, load_from_csv
from .data_utils import write_to_csv

from .data_utils import build_graph
from .data_utils import src_T_dst
from .data_utils import get_triple_set
from .data_utils import get_all_triples
from .data_utils import get_select_src_rate

# os.environ['KG_DIR'] = os.path.join(os.path.expanduser('~'), '.KGDataSets')

