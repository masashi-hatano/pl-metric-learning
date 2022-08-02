import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(Cifar10DataModule, self).__init__()

        self.cfg = config.data_module
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                self.cfg.mean, 
                self.cfg.std, 
            ),
        ])
    
    def prepare_data(self) -> None:
        self.train = datasets.CIFAR10(root=self.cfg.data_dir, train=True, download=True, transform=self.transform)
        self.testset = datasets.CIFAR10(root=self.cfg.data_dir, train=False, download=False, transform=self.transform)
    
    def setup(self, stage=None):
        n_train = int(len(self.train)*0.8)
        n_val = len(self.train)-n_train
        self.cifar10_train, self.cifar10_val = random_split(self.train, [n_train, n_val])
        self.cifar10_test = self.testset

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)