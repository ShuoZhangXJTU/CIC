import torch
import sklearn.metrics as metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluation_indicators(mode, tsx_writer, pred, target, cur_loss, ttl_step):
    """
    :param ttl_step: step for 'train' mode, epoch for 'val' mode
    :param cur_loss:
    :param tsx_writer: tensorboardx writer
    :param mode: str train, val or test
    :param pred: 0/1 tensor of type long
    :param target: 0/1 tensor of type long
    """
    y_score, y_pred, y_true = pred[1].cpu().numpy(), pred[1].cpu().gt(0.5).long().numpy(), target.cpu().numpy()

    # -- base metrics
    balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average='binary', zero_division=0)
    F1 = metrics.f1_score(y_true, y_pred, average='binary', zero_division=0)
    MCC = metrics.matthews_corrcoef(y_true, y_pred)
    AUC = metrics.roc_auc_score(y_true, y_score, average='weighted', labels=[0, 1])

    # -- curves
    precision_PR, recall_PR, thresholds_PR = metrics.precision_recall_curve(y_true, y_score)
    # PR_display = metrics.PrecisionRecallDisplay(precision=precision_PR, recall=recall_PR).plot()
    precision_ROC, recall_ROC, thresholds_ROC = metrics.roc_curve(y_true, y_score)
    # roc_display = metrics.RocCurveDisplay(fpr=precision_ROC, tpr=recall_ROC).plot()
    if mode != 'tune':
        tsx_writer.add_scalar('{}/Loss'.format(mode), cur_loss, ttl_step)
        tsx_writer.add_scalar('{}/Acc'.format(mode), balanced_accuracy, ttl_step)
        tsx_writer.add_scalar('{}/Precision'.format(mode), precision, ttl_step)
        tsx_writer.add_scalar('{}/Recall'.format(mode), recall, ttl_step)
        tsx_writer.add_scalar('{}/F1'.format(mode), F1, ttl_step)
        tsx_writer.add_scalar('{}/MCC'.format(mode), MCC, ttl_step)
        tsx_writer.add_scalar('{}/AUC'.format(mode), AUC, ttl_step)

    return balanced_accuracy, precision, recall, F1, MCC, AUC
