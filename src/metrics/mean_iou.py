import torch


def mean_iou_bbox(boxes_1: torch.Tensor, boxes_2: torch.Tensor) -> torch.Tensor:
    """Compute the IoU of the cartesian product of two sets of boxes.

    Each box in each set shall be (x1, y1, x2, y2).

    Args:
        boxes_1: a tensor of bounding boxes in :math:`(B1, 4)`.
        boxes_2: a tensor of bounding boxes in :math:`(B2, 4)`.

    Returns:
        a tensor in dimensions :math:`(B1, B2)`, representing the
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2.

    Example:
        >> boxes_1 = torch.tensor([[40, 40, 60, 60], [30, 40, 50, 60]])
        >> boxes_2 = torch.tensor([[40, 50, 60, 70], [30, 40, 40, 50]])
        >> mean_iou_bbox(boxes_1, boxes_2)
        tensor([[0.3333, 0.0000],
                [0.1429, 0.2500]])
    """
    # TODO: support more box types. e.g. xywh,
    if not (((boxes_1[:, 2] - boxes_1[:, 0]) >= 0).all() or ((boxes_1[:, 3] - boxes_1[:, 1]) >= 0).all()):
        raise AssertionError("Boxes_1 does not follow (x1, y1, x2, y2) format.")
    if not (((boxes_2[:, 2] - boxes_2[:, 0]) >= 0).all() or ((boxes_2[:, 3] - boxes_2[:, 1]) >= 0).all()):
        raise AssertionError("Boxes_2 does not follow (x1, y1, x2, y2) format.")
    # find intersection
    lower_bounds = torch.max(boxes_1[:, :2].unsqueeze(1), boxes_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(boxes_1[:, 2:].unsqueeze(1), boxes_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (boxes_1[:, 2] - boxes_1[:, 0]) * (boxes_1[:, 3] - boxes_1[:, 1])  # (n1)
    areas_set_2 = (boxes_2[:, 2] - boxes_2[:, 0]) * (boxes_2[:, 3] - boxes_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)
