# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm


class Trainer(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_epoch=1000,
                 learning_rate=0.5,
                 use_gpu=True,
                 opt_method="SGD",
                 scheduler_step=100,
                 scheduler_gamma=0.75,
                 save_steps=None,
                 checkpoint_dir=None):
        '''args:
        model: Model
        data_loader: The data loader for training
        train_epoch: Epoch number
        learning_rate: learning rate
        use_gpu: Use GPU, default: True
        opt_method: optime method, default: "SGD", or "Adam", "Adagrad", "Adadelta".
        scheduler_step: the scheduler step
        scheduler gamma: the gamma in StepLR
        save_steps: save model per n steps.
        checkpoint_dir: the checkpoint dir.
        '''

        self.work_threads = 8
        self.train_epoch = train_epoch

        self.opt_method = opt_method
        self.optimizer = None
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.lr_decay = 0
        self.weight_decay = 0
        self.learning_rate = learning_rate

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.learning_rate,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step,
                                              gamma=self.scheduler_gamma, last_epoch=-1)
        print("Finish initializing...")

        training_range = tqdm(range(self.train_epoch))
        for epoch in training_range:
            res = 0.0
            for data in self.data_loader:
                loss = self.train_one_step(data)
                res += loss
            training_range.set_description(
                "Epoch %d | loss: %f" % (epoch, res))
            scheduler.step()
            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(
                    self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_epoch(self, train_epoch):
        self.train_epoch = train_epoch

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
