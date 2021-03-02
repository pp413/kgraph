#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2021/02/26 22:26:00
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#

from ._utils import load_from_text, load_from_csv
from ._utils import write_to_csv
from ._utils import get_triple_set, get_all_triples
from ._utils import get_select_src_rate
from ._utils import src_T_dst, build_graph

from ._data import load_fb15k, load_fb15k237
from ._data import load_wn18, load_wn18rr
from ._data import DataBase
from ._data import FB15k, FB15k237
from ._data import WN18, WN18RR

from ._downloading import single_thread_download, ManyThreadDownload
