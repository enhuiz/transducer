from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Function

import transducer_cpp


class Transducer(Function):
    @staticmethod
    def forward(ctx, log_probs, labels, lengths, label_lengths, blank=None):
        """
        Computes the Transducer cost for a minibatch of examples.

        Arguments:
            log_probs (FloatTensor): The log probabilities should
                be of shape
                (minibatch, input len, output len, vocab size).
            labels (IntTensor): 2D tensor of labels for each example
                consecutively.
            lengths (IntTensor): 1D tensor of number actviation time-steps
                for each example.
            label_lengths (IntTensor): 1D tensor of label lengths for
                each example.

        Returns:
            costs (FloatTensor): .
        """
        is_cuda = log_probs.is_cuda

        certify_inputs(log_probs, labels, lengths, label_lengths)

        log_probs = log_probs.cpu()
        costs = torch.zeros(log_probs.shape[0])
        grads = log_probs.new(log_probs.shape).zero_()

        if blank is None:
            blank = log_probs.shape[-1] - 1

        transducer_cpp.transduce(log_probs, labels,
                                 lengths, label_lengths,
                                 costs, grads, blank)
        if is_cuda:
            costs = costs.cuda()
            grads = grads.cuda()
        ctx.save_for_backward(grads)

        return costs

    @staticmethod
    def backward(ctx, cost):
        return ctx.saved_tensors[0], None, None, None, None


class TransducerLoss(nn.Module):
    def __init__(self, blank=None, reduction='mean'):
        super().__init__()
        self.blank = blank
        self.reduction = reduction

    def forward(self, log_probs, labels, lengths, label_lengths):
        """
        Args:
            log_probs: (N, T, S, D)
            labels: (N, L)
            lengths: (N)
            label_lengths: (N)
        """
        labels = [i for label, label_length in zip(labels, label_lengths)
                  for i in label[:label_length]]
        labels = torch.tensor(labels)
        loss = Transducer.apply(log_probs,
                                labels,
                                lengths,
                                label_lengths,
                                self.blank)
        if self.reduction == 'sum':
            loss = loss / label_lengths
        loss = loss.mean()

        return loss


def check_type(var, t, name):
    if var.dtype is not t:
        raise TypeError("{} must be {}".format(name, t))


def check_contiguous(var, name):
    if not var.is_contiguous():
        raise ValueError("{} must be contiguous".format(name))


def check_dim(var, dim, name):
    if len(var.shape) != dim:
        raise ValueError("{} must be {}D".format(name, dim))


def certify_inputs(log_probs, labels, lengths, label_lengths):
    check_type(log_probs, torch.float32, "log_probs")
    check_type(labels, torch.int32, "labels")
    check_type(label_lengths, torch.int32, "label_lengths")
    check_type(lengths, torch.int32, "lengths")
    check_contiguous(labels, "labels")
    check_contiguous(label_lengths, "label_lengths")
    check_contiguous(lengths, "lengths")

    if lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("must have a length per example.")
    if label_lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("must have a label length per example.")

    check_dim(log_probs, 4, "log_probs")
    check_dim(labels, 1, "labels")
    check_dim(lengths, 1, "lenghts")
    check_dim(label_lengths, 1, "label_lenghts")
    max_T = torch.max(lengths)
    max_U = torch.max(label_lengths)
    T, U = log_probs.shape[1:3]
    if T != max_T:
        raise ValueError("Input length mismatch")
    if U != max_U + 1:
        raise ValueError("Output length mismatch")
