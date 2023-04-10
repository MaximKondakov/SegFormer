def category_ids():
    """
    Label id initialization
    :return: dict of the {name : id_name}
    """
    category_ids = {
        'background': 0,
        'apples': 1,
        'paints': 2,
        'teapots': 3,
        'clothes': 4,
        'pork_fat': 5
    }  # category_dict
    return category_ids


# make labels
def make_labels():
    label2id = category_ids()
    num_labels = len(label2id)
    id2label = {v: k for k, v in label2id.items()}
    return num_labels, id2label, label2id


def color_map():
    """
    Color map initialization
    :return: dict of the {id : color}
    """
    color_map = {
        0: (0, 0, 0),  # background
        1: (216, 82, 24),  # apples
        2: (111, 111, 211),  # paints
        3: (125, 46, 141),  # teapots
        4: (118, 171, 47),  # clothes
        5: (161, 19, 46),  # pork_fat
    }
    return color_map
