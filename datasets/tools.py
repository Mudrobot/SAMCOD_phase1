import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from segment_anything.utils.transforms import ResizeLongestSide
from imagecorruptions import corrupt, get_corruption_names

# A.RandomCropNearBBox()
# A.BBoxSafeRandomCrop()
# A.RandomSizedBBoxSafeCrop()

# 定义一个图像增强管道，其中包含了一些简单的变换，使用albumentations库的Compose将多个变换组合在一起
weak_transforms = A.Compose(
    [
        # 随机翻转图像，这种变换不会改变图像的内容，但会使图像变得不同
        A.Flip(),

        # 水平翻转图像，用于模拟不同的视角和方向
        A.HorizontalFlip(),

        # 垂直翻转图像，同样是用于视角和方向的变换
        A.VerticalFlip(),

        # 注释掉的部分：可以选择使用该变换来裁剪图像，但它被禁用了
        # A.BBoxSafeRandomCrop()  # 在保持目标框完整性的情况下，随机裁剪图像
    ],
    # bbox_params 设定了与边界框相关的参数，格式为pascal_voc，标注字段为category_ids
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),

    # 注释掉的部分：keypoint_params 用于处理关键点标注的格式，这里被禁用了
    # keypoint_params=A.KeypointParams(format='xy')
)


strong_transforms = A.Compose(
    [
        # 将图像的颜色深度减少，通常用于创造抽象效果或插图风格
        A.Posterize(),

        # 对图像进行直方图均衡化，改善图像的对比度
        A.Equalize(),

        # 对图像进行锐化处理，增强细节，使图像的边缘更加明显
        A.Sharpen(),

        # 模拟太阳化效果，将高亮部分反转，创造类似负片效果
        A.Solarize(),

        # 随机调整图像的亮度和对比度，增强图像的多样性
        A.RandomBrightnessContrast(),

        # 随机在图像中添加阴影，模拟光照变化，增加真实感
        A.RandomShadow(),
    ]
)

class WOResizeAndPad:
    def __init__(self):
        # 不再需要 target_size 或 transform
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes=None, visual=False):
        # 不做resize和pad，直接转换为张量
        image = self.to_tensor(image)
        masks = [self.to_tensor(mask) for mask in masks]

        if bboxes is not None:
            # 处理边界框（如有提供）
            bboxes = [
                [bbox[0], bbox[1], bbox[2], bbox[3]]  # 保持原始坐标
                for bbox in bboxes
            ]
            if visual:
                return image, masks, bboxes
            else:
                return image, masks, bboxes
        else:
            if visual:
                return image, masks
            else:
                return image, masks

    def transform_image(self, image):
        # 直接转换为张量，不做resize和pad
        return self.to_tensor(image)

    def transform_coord(self, points, image):
        # 不做resize和pad，直接返回原始坐标
        return points

    def transform_coords(self, points, image, n):
        # 同样不做resize和pad，直接返回原始坐标
        return points

class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes=None, visual=False):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        if bboxes is not None:
            bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
            bboxes = [
                [bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h]
                for bbox in bboxes
            ]
            if visual:
                return padding, image, masks, bboxes
            else:
                return image, masks, bboxes
        else:
            if visual:
                return padding, image, masks
            else:
                return image, masks

    def transform_image(self, image):
        # Resize image and masks
        image = self.transform.apply_image(image)
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        return image

    def transform_coord(self, points, image):
        og_h, og_w, _ = image.shape
        coords = points.reshape(1, -1, 2)
        points = self.transform.apply_coords(coords, (og_h, og_w))
        return points.reshape(-1, 2)

    def transform_coords(self, points, image, n):
        og_h, og_w, _ = image.shape
        coords = points.reshape(-1, n, 2)
        points = self.transform.apply_coords(coords, (og_h, og_w))
        return points.reshape(-1, n, 2)


def corrupt_image(image, filename):
    file_name = os.path.basename(os.path.abspath(filename))
    file_path = os.path.dirname(os.path.abspath(filename))
    for corruption in get_corruption_names():
        corrupted = corrupt(image, severity=5, corruption_name=corruption)
        corrupt_path = file_path.replace(
            "val2017", os.path.join("corruption", corruption)
        )
        if not os.path.exists(corrupt_path):
            os.makedirs(corrupt_path, exist_ok=True)
        cv2.imwrite(os.path.join(corrupt_path, file_name), corrupted)


def soft_transform(image: np.ndarray, bboxes: list, masks: list, categories: list):
    weak_transformed = weak_transforms(
        image=image, bboxes=bboxes, masks=masks, category_ids=categories
    )
    image_weak = weak_transformed["image"]
    bboxes_weak = weak_transformed["bboxes"]
    masks_weak = weak_transformed["masks"]

    strong_transformed = strong_transforms(image=image_weak)
    image_strong = strong_transformed["image"]
    return image_weak, bboxes_weak, masks_weak, image_strong


def soft_transform_all(
    image: np.ndarray, bboxes: list, masks: list, points: list, categories: list
):
    weak_transformed = weak_transforms(
        image=image,
        bboxes=bboxes,
        masks=masks,
        category_ids=categories,
        keypoints=points,
    )
    image_weak = weak_transformed["image"]
    bboxes_weak = weak_transformed["bboxes"]
    masks_weak = weak_transformed["masks"]
    keypoints_weak = weak_transformed["keypoints"]

    strong_transformed = strong_transforms(image=image_weak)
    image_strong = strong_transformed["image"]
    return image_weak, bboxes_weak, masks_weak, keypoints_weak, image_strong


def collate_fn(batch):
    if len(batch[0]) == 4:
        images, bboxes, masks, path = zip(*batch)
        images = torch.stack(images)
        return images, bboxes, masks, path
    elif len(batch[0]) == 5:
        images_soft, images, bboxes, masks, path = zip(*batch)
        images = torch.stack(images)
        images_soft = torch.stack(images_soft)
        return images_soft, images, bboxes, masks, path
    else:
        raise ValueError("Unexpected batch format")


def collate_fn_(batch):
    return zip(*batch)


def decode_mask(mask):
    """
    Convert mask with shape [1, h, w] using 1, 2, 3, ... to represent different objects
    to a mask with shape [n, h, w] using a new dimension to represent the number of objects.

    Args:
        mask (torch.Tensor): Mask tensor with shape [1, h, w] using 1, 2, 3, ... to represent different objects.

    Returns:
        torch.Tensor: Mask tensor with shape [n, h, w] using a new dimension to represent the number of objects.
    """
    unique_labels = torch.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    n_objects = len(unique_labels)
    new_mask = torch.zeros((n_objects, *mask.shape[1:]), dtype=torch.int64)
    for i, label in enumerate(unique_labels):
        new_mask[i] = (mask == label).squeeze(0)
    return new_mask


def encode_mask(mask):
    """
    Convert mask with shape [n, h, w] using a new dimension to represent the number of objects
    to a mask with shape [1, h, w] using 1, 2, 3, ... to represent different objects.

    Args:
        mask (torch.Tensor): Mask tensor with shape [n, h, w] using a new dimension to represent the number of objects.

    Returns:
        torch.Tensor: Mask tensor with shape [1, h, w] using 1, 2, 3, ... to represent different objects.
    """
    n_objects = mask.shape[0]
    new_mask = torch.zeros((1, *mask.shape[1:]), dtype=torch.int64)
    for i in range(n_objects):
        new_mask[0][mask[i] == 1] = i + 1
    return new_mask


if __name__ == "__main__":
    mask_encode = np.array([[[0, 0, 1], [2, 0, 2], [0, 3, 3]]])
    mask_decode = np.array(
        [
            [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 1, 1]],
        ]
    )
    encoded_mask = encode_mask(torch.tensor(mask_decode))
    decoded_mask = decode_mask(torch.tensor(mask_encode))
