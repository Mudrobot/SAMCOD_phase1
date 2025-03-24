from datasets.CODAll import load_datasets

class CFG:
    def __init__(self):
        self.batch_size = 4
        self.num_workers = 4
        self.augment = True
        self.val_batchsize = 4
        self.get_prompt = False
        
cfg = CFG()
train, val = load_datasets(cfg, 1024)