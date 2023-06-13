import torch
import numpy as np
from torch import nn


class PadSequence(object):

    def __init__(self, max_length, pad_token, prediction_start_hour, prediction_length):
        self.max_length = max_length
        self.PAD_token = pad_token
        self.prediction_start_hour = prediction_start_hour
        self.trim_input = prediction_start_hour + prediction_length

    def __call__(self, received_sample):
        sample, labels = received_sample
        sample = torch.Tensor(sample)
        labels = torch.Tensor(labels)

        sample = sample[:self.trim_input]
        labels = labels[:self.trim_input]
        sample = torch.reshape(sample, (sample.size()[0], 1, sample.size()[1], sample.size()[2]))

        return sample, labels

class PadSequenceNoTrim(object):

    def __init__(self, max_length, pad_token, prediction_start_hour, prediction_length):
        self.max_length = max_length
        self.PAD_token = pad_token
        self.prediction_start_hour = prediction_start_hour
        self.trim_input = prediction_start_hour + prediction_length

    def __call__(self, received_sample):
        sample, labels = received_sample
        sample = torch.Tensor(sample)
        labels = torch.Tensor(labels)

        sample = sample
        labels = labels
        sample = torch.reshape(sample, (sample.size()[0], 1, sample.size()[1], sample.size()[2]))

        return sample, labels

