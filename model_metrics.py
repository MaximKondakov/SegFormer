import evaluate
import torch
from torch import nn
from config_category_label_and_color import make_labels


def compute_metrics(eval_pred):
    """
    Compute training / validation metrics.
    This is called by the trainer.
    :param eval_pred:
    :return: metrics per category and mean value
    """

    metric = evaluate.load("mean_iou")
    num_labels, id2label, label2id = make_labels()
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics = metric._compute(
        predictions=pred_labels,
        references=labels,
        num_labels=num_labels,
        ignore_index=0,
        reduce_labels=False,
    )
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()
    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})  # wrong iou

    return metrics


def main():
    return compute_metrics


if __name__ == "__main__":
    main()
