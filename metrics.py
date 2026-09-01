from torchmetrics.classification import (BinaryF1Score, BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinarySpecificity,
                                         BinaryRecall, BinaryNegativePredictiveValue, MulticlassF1Score,
                                         BinaryAveragePrecision, BinaryMatthewsCorrCoef, BinaryROC,
                                         BinaryPrecisionRecallCurve, BinaryConfusionMatrix)
import torch


class Metrics():

    """ Clase que calcula las metricas de evaluacion de similitud entre dos imagenes """

    def __init__(self, device, neg_weight, pos_weight) -> None:

        self.accuracy       = BinaryAccuracy(threshold=0.5).to(device)
        self.sensitivity    = BinaryRecall().to(device)
        self.specificity    = BinarySpecificity().to(device)
        self.ppv            = BinaryPrecision().to(device)
        self.npv            = BinaryNegativePredictiveValue().to(device)
        self.f1_score       = BinaryF1Score().to(device)
        self.f1_score_macro = MulticlassF1Score(num_classes=2, average="macro").to(device)
        self.auc            = BinaryAUROC().to(device)
        self.auprc          = BinaryAveragePrecision().to(device)
        self.mcc            = BinaryMatthewsCorrCoef().to(device)
        self.roc_curve      = BinaryROC().to(device)
        self.pr_curve       = BinaryPrecisionRecallCurve().to(device)
        self.confusion      = BinaryConfusionMatrix().to(device)
        self.bce_loss       = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weight, pos_weight])).to(device)

        # Los objetos de torchmetrics acumulan estado entre llamadas, así que se
        # guardan juntos para poder reiniciarlos antes de cada cálculo.
        self.torchmetrics = (
            self.accuracy, self.sensitivity, self.specificity, self.ppv, self.npv,
            self.f1_score, self.f1_score_macro, self.auc, self.auprc, self.mcc,
            self.roc_curve, self.pr_curve, self.confusion
        )

    def get_metrics(self, prediction: torch.tensor, target: torch.tensor, probs: torch.tensor, stage: str)->dict:

        """
        Calculate metrics for a given set of predictions and targets.

        Las métricas se calculan sobre el conjunto completo que se recibe, no lote
        a lote: AUC, AU-PRC y MCC no tienen sentido promediados por lote.

        Args:
            prediction (torch.tensor): Predictions tensor.
            target (torch.tensor): Target tensor.
            probs (torch.tensor): Probability tensor.
            stage (str): Prefix of the metric names.

        Returns:
            dict: Dictionary of metrics.
        """

        self.reset()

        # Calcular las metricas de clasificacion
        accuracy_       = self.accuracy(prediction, target)
        sensitivity_    = self.sensitivity(prediction, target)
        specificity_    = self.specificity(prediction, target)
        ppv_            = self.ppv(prediction, target)
        npv_            = self.npv(prediction, target)
        f1_score_       = self.f1_score(prediction, target)
        f1_score_macro_ = self.f1_score_macro(prediction, target)
        auc_            = self.auc(probs[:, 1], target)
        auprc_          = self.auprc(probs[:, 1], target)
        mcc_            = self.mcc(prediction, target)
        bce_error_      = self.bce_loss(probs, target)

        dict_metrics = {
            f"{stage}_Accuracy"       : accuracy_,
            f"{stage}_Sensitivity"    : sensitivity_,
            f"{stage}_Specificity"    : specificity_,
            f"{stage}_PPV"            : ppv_,
            f"{stage}_NPV"            : npv_,
            f"{stage}_F1-Score"       : f1_score_,
            f"{stage}_F1-Score-Macro" : f1_score_macro_,
            f"{stage}_AUC"            : auc_,
            f"{stage}_AU-PRC"         : auprc_,
            f"{stage}_MCC"            : mcc_,
            f"{stage}_BCE-Loss"       : bce_error_
        }

        return dict_metrics

    def get_curves(self, target: torch.tensor, probs: torch.tensor)->dict:

        """
        Calculate the ROC and the precision-recall curves of a set of samples.

        Args:
            target (torch.tensor): Target tensor.
            probs (torch.tensor): Probability tensor.

        Returns:
            dict: Points of both curves as numpy arrays.
        """

        self.reset()

        fpr, tpr, _             = self.roc_curve(probs[:, 1], target)
        precision, recall, _    = self.pr_curve(probs[:, 1], target)

        dict_curves = {
            "fpr"       : fpr.cpu().numpy(),
            "tpr"       : tpr.cpu().numpy(),
            "precision" : precision.cpu().numpy(),
            "recall"    : recall.cpu().numpy()
        }

        return dict_curves

    def get_confusion_matrix(self, prediction: torch.tensor, target: torch.tensor)->torch.tensor:

        """
        Calculate the confusion matrix of a set of predictions.

        Args:
            prediction (torch.tensor): Predictions tensor.
            target (torch.tensor): Target tensor.

        Returns:
            torch.tensor: 2x2 matrix ordered as [[TN, FP], [FN, TP]].
        """

        self.reset()

        return self.confusion(prediction, target)

    def reset(self) -> None:

        """
        Reset the internal state of every torchmetrics object.

        Returns:
            None
        """

        for metric in self.torchmetrics:
            metric.reset()
