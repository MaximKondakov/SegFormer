from transformers import SegformerForSemanticSegmentation
from config_category_label_and_color import make_labels


def get_model(model_name):
    """
    Load pretrained model from huggingface
    :param model_name: name of the model
    :return: initialized model
    """
    pretrained_model_name = model_name
    num_labels, id2label, label2id = make_labels()
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        reshape_last_stage=True,

    )

    return model


def main():
    return get_model(model_name=None)


if __name__ == '__main__':
    main()
