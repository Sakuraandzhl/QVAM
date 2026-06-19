# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import heapq
import atexit
import bisect
from collections import deque

import cv2
import torch
import torch.multiprocessing as mp
import numpy as np
from fastreid.engine import DefaultPredictor

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, list):
        if len(x) == 0:
            return np.array([])
        if isinstance(x[0], torch.Tensor):
            return torch.stack(x).cpu().numpy()
        return np.array(x)
    else:
        return np.array(x)

class FeatureExtractionDemo(object):
    def __init__(self, cfg, parallel=False):
        self.cfg = cfg
        self.parallel = parallel

        if parallel:
            # 多GPU异步模式
            self.num_gpus = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, self.num_gpus)
        else:
            # 单进程模式
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, original_image, camid, viewid):
        # BGR转RGB
        original_image = original_image[:, :, ::-1]
        # 调整尺寸
        #cv2默认图像尺寸为[宽，高]因此要反转
        #INTER_CUBIC双三次插值，效果好但较慢
        image = cv2.resize(original_image, tuple(self.cfg.INPUT.SIZE_TEST[::-1]),
                           interpolation=cv2.INTER_CUBIC)
        # 转换为张量并添加批次维度
        #opencv图像为[h,w,c]模型输入为[c,h,w]
        #opencv读取图像格式为numpy.ndarray,要转为torch.tensor
        #as_tensor 会尽量与原 numpy 数组共享内存，效率高
        #[none]在最前面添加批次维度变为[1,3,256,128]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
        # 提取特征
        predictions = self.predictor(image, camid, viewid)
        return predictions

    def run_on_loader(self, data_loader):
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size  # 缓冲区大小

            batch_data = deque()  # 用于保存批次数据的双端队列   生产者-消费者模型

            for cnt, batch in enumerate(data_loader):#cnt批次索引 batch批次数据
                # 1. 保存原始批次数据
                batch_data.append(batch)

                # 2. 将图片放入预测器队列（异步处理）
                self.predictor.put(batch["images"])

                # 3. 当队列达到缓冲大小时，开始取出结果
                if cnt >= buffer_size:
                    batch = batch_data.popleft()  # 取出最早的批次数据
                    predictions = self.predictor.get()  # 获取对应结果

                    yield predictions, to_numpy(batch["targets"]), to_numpy(batch["camids"]), to_numpy(batch["viewids"])
            # 4. 处理剩余的批次
            while len(batch_data):
                batch = batch_data.popleft()
                predictions = self.predictor.get()
                yield predictions, to_numpy(batch["targets"]), to_numpy(batch["camids"]), to_numpy(batch["viewids"])
        else:
            for batch in data_loader:
                predictions = self.predictor(batch["images"], batch["camids"], batch["viewids"])
                yield predictions, to_numpy(batch["targets"]), to_numpy(batch["camids"]), to_numpy(batch["viewids"])
                #yield返回结果而不结束函数，下次调用继续从这里执行
"""
AsyncPredictor相当于一座工厂，而_PredictWorker是里面的工人(GPU数量/进程数量)
FeatureExtractionDemo是boss，收到data_loader订单，通过run_on_loader送入工厂
工厂的


"""
class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because when the amount of data is large.
    """

    class _StopToken:#一个空的标记类，用于通知工作进程停止运行。当把 _StopToken() 放入任务队列时，工作进程收到后会退出循环
        pass

    class _PredictWorker(mp.Process):#继承 可以被start()启动 每个GPU对应一个进程！！！
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """

        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)#进程数量
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        #为什么是三倍
        #太少工作进程频繁等待，GPU空转
        #太多堆积图片占用内存
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                self._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []
        #self.result_heap = []
        for p in self.procs:
            p.start()

        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))
        
    def get(self):
        self.get_idx += 1
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # Make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)
    #堆优化 O(logn)
    # def get1(self):
    #     self.get_idx += 1
    #     if len(self.result_heap) and self.result_heap[0][0] == self.get_idx:
    #         res = self.result_heap[0][1]
    #         heapq.heappop(self.result_heap)
    #         return res
    #     while True:
    #         # Make sure the results are returned in the correct order
    #         idx, res = self.result_queue.get()
    #         if idx == self.get_idx:
    #             return res
    #         heapq.heappush(self.result_heap, (idx, res))

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
