#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: https://www.pythonf.cn/read/112198
# @ Date: 2021/02/28 21:39:16
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
import os, requests, time, threading
from queue import Queue
from tqdm import tqdm


# lock = threading.Lock()           # 改成局部变量写文件,就无需线程锁
# ===================================================================================================================


# 单线程下载文件,download_from_url(url,"./aa.mp3")
def single_thread_download(url, dst):
        """
        @param: url to download file
        @param: dst place to put the file
        """
        #file_size = int(urlopen(url).info().get('Content-Length', -1))
        file_size = int(requests.head(url).headers['Content-Length'])
        if os.path.exists(dst):
            first_byte = os.path.getsize(dst)
        else:
            first_byte = 0
        if first_byte >= file_size:
            return file_size
        header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
        pbar = tqdm(
            total=file_size, initial=first_byte,
            unit='B', unit_scale=True, desc=url.split('/')[-1])
        req = requests.get(url, headers=header, stream=True)
        with(open(dst, 'ab')) as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(1024)
        pbar.close()
        return file_size


class ManyThreadDownload:
    def __init__(self, num=10):
        self.num = num              # 线程数,默认10
        self.url = ''               # url
        self.name = ''              # 目标地址
        self.total = 0              # 文件大小

    # 获取每个线程下载的区间
    def get_range(self):
        ranges = []
        offset = int(self.total/self.num)
        for i in range(self.num):
            if i == self.num-1:
                ranges.append((i*offset, ''))
            else:
                ranges.append(((i * offset), (i + 1) * offset - 1))
        return ranges               # [(0,99),(100,199),(200,"")]

    # 通过传入开始和结束位置来下载文件
    def download(self, ts_queue):
        while not ts_queue.empty():
            start_, end_ = ts_queue.get()
            headers = {
                'Range': 'Bytes=%s-%s' % (start_, end_),
                'Accept-Encoding': '*'
                }
            flag = False
            while not flag:
                try:
                    # 设置重连次数
                    requests.adapters.DEFAULT_RETRIES = 10
                    # s = requests.session()            # 每次都会发起一次TCP握手,性能降低，还可能因发起多个连接而被拒绝
                    # # 设置连接活跃状态为False
                    # s.keep_alive = False
                    # 默认stream=false,立即下载放到内存,文件过大会内存不足,大文件时用True需改一下码子
                    res = requests.get(self.url, headers=headers)
                    res.close()                         # 关闭请求  释放内存
                except Exception as e:
                    print((start_, end_, "出错了,连接重试:%s", e, ))
                    time.sleep(1)
                    continue
                flag = True

            print("\r", ("%s-%s download success" % (start_, end_)), end="", flush=True)
            # with lock:
            with open(self.name, "rb+") as fd:
                fd.seek(start_)
                fd.write(res.content)
            # self.fd.seek(start_)                                        # 指定写文件的位置,下载的内容放到正确的位置处
            # self.fd.write(res.content)                                  # 将下载文件保存到 fd所打开的文件里

    def run(self, url, name):
        self.url = url
        self.name = name
        self.total = int(requests.head(url).headers['Content-Length'])
        # file_size = int(urlopen(self.url).info().get('Content-Length', -1))
        file_size = self.total
        if os.path.exists(name):
            first_byte = os.path.getsize(name)
        else:
            first_byte = 0
        if first_byte >= file_size:
            return file_size

        self.fd = open(name, "wb")                   # 续传时直接rb+ 文件不存在时会报错,先wb再rb+
        self.fd.truncate(self.total)                 # 建一个和下载文件一样大的文件,不是必须的,stream=True时会用到
        self.fd.close()
        # self.fd = open(self.name, "rb+")           # 续传时ab方式打开时会强制指针指向文件末尾,seek并不管用,应用rb+模式
        thread_list = []
        ts_queue = Queue()                           # 用队列的线程安全特性，以列表的形式把开始和结束加到队列
        for ran in self.get_range():
            start_, end_ = ran
            ts_queue.put((start_, end_))

        for i in range(self.num):
            t = threading.Thread(target=self.download, name='th-' + str(i), kwargs={'ts_queue': ts_queue})
            t.setDaemon(True)
            thread_list.append(t)
        for t in thread_list:
            t.start()
        for t in thread_list:
            t.join()                                # 设置等待，全部线程完事后再继续

        self.fd.close()
