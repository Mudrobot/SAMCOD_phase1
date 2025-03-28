from datasets.CODAll import load_datasets, CODAllDataset
from datasets.tools import WOResizeAndPad, ResizeAndPad, soft_transform, collate_fn, collate_fn_, decode_mask


class CFG:
    def __init__(self):
        self.batch_size = 4
        self.num_workers = 4
        self.augment = True
        self.val_batchsize = 4
        self.get_prompt = False
        
cfg = CFG()
# train, val = load_datasets(cfg, 1024)

transform = ResizeAndPad(1024)

train = CODAllDataset(
    cfg,
    image_root='./data/CODAll/',
    gt_root='./data/CODAll/',
    transform=transform,
    # split=cfg.split,
    training=True,
    if_self_training=cfg.augment,
)

val = CODAllDataset(
    cfg,
    image_root='./data/CODAll/',
    gt_root='./data/CODAll/',
    transform=transform,
    eval_dataset='cham'
    # split=cfg.split,
)
print(len(val))
# print(train[3046][-2].shape)
# for i in range(len(train)):
#     if 'camourflage_00007' in train[i][-1]:
#         print(i)
#         break