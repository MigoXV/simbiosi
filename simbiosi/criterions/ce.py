import math
from dataclasses import dataclass

import numpy as np
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

# 为了避免除 0 的问题，加入一个极小值 epsilon
EPS = 1e-8


@register_criterion("simbiosi_ce")
class CrossEntropyLoss(FairseqCriterion):

    def __init__(self, task):
        super().__init__(task)
        self.num_classes = task.num_classes

    def forward(self, model, sample, reduce=True):
        # 假设 sample 为 (features, targets)，其中 targets 为类别索引
        features, targets = sample
        outputs = model(features)  # 输出 shape: [batch_size, num_classes]
        # 多分类交叉熵损失（内部会先对输出做 softmax）
        loss = torch.nn.functional.cross_entropy(
            outputs, targets, reduction="mean" if reduce else "none"
        )
        sample_size = targets.numel()
        # 计算预测结果（取最大概率对应的类别）
        targets = targets.detach().cpu().numpy()
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        # 算混淆矩阵
        confusion_matrix = np.bincount(
            targets * self.num_classes + preds,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        # confusion_matrix = confusion_matrix.detach()
        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size,
            "nsentences": sample_size,
            "sample_size": sample_size,
            "confusion_matrix": confusion_matrix,
        }
        return loss, sample_size, logging_output

    def logging_outputs_can_be_summed(self):
        return True

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        loss_mean = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        # loss,nsentences,ntokens
        metrics.log_scalar(
            "loss",
            loss_mean,
            sample_size,
            round=3,
            priority=1,
        )
        metrics.log_scalar("nsentences", nsentences, sample_size, round=3)
        metrics.log_scalar("ntokens", ntokens, sample_size, round=3)
        # 计算混淆矩阵
        confusion_matrix = (
            sum(log.get("confusion_matrix", 0) for log in logging_outputs)
        )
        metrics.log_scalar_sum(
            "confusion_matrix", confusion_matrix, sample_size, round=3
        )
        metrics.log_derived("a", cls.derive_accurency, priority=2)
        metrics.log_derived("macro_p", cls.derive_macro_precision, priority=4)
        metrics.log_derived("macro_r", cls.derive_macro_recall, priority=5)
        metrics.log_derived("macro_f1", cls.derive_macro_f1, priority=3)

    @staticmethod
    def derive_accurency(meter_dict):
        confuse_matrix = meter_dict["confusion_matrix"].smoothed_value
        acc = np.trace(confuse_matrix) / confuse_matrix.sum()
        return round(float(acc * 100), 1)

    @staticmethod
    def derive_macro_precision(meter_dict):
        confusion_matrix = meter_dict["confusion_matrix"].smoothed_value
        p = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + EPS)
        p = round(float(p.mean() * 100), 1)
        return p

    @staticmethod
    def derive_macro_recall(meter_dict):
        confusion_matrix = meter_dict["confusion_matrix"].smoothed_value
        r = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + EPS)
        r = round(float(r.mean() * 100), 1)
        return r

    @staticmethod
    def derive_macro_f1(meter_dict):
        confusion_matrix = meter_dict["confusion_matrix"].smoothed_value
        tp = np.diag(confusion_matrix)
        fp = np.sum(confusion_matrix, axis=0) - tp
        fn = np.sum(confusion_matrix, axis=1) - tp
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1_scores = 2 * (precision * recall) / (precision + recall + EPS)
        return round(float(np.mean(f1_scores) * 100), 1)
