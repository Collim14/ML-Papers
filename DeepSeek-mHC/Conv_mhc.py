import torch
import torch.nn as nn
from mhc_wrap import mHCWrapper, MHCEntry, MHCExit
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class MNISTModel(nn.Module):
    def __init__(self, n_streams=4, num_classes=10,mhc = False,useNS=False, static = False):
        super().__init__()
        self.mhc = mhc
        self.n_streams = n_streams if mhc else 1
        self.num_classes = num_classes
        self.useNS = useNS
        self.static = static
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.entry = MHCEntry(self.n_streams) if mhc else nn.Identity()

        self.backbone = nn.Sequential(
            self._wrap(
                nn.Sequential(nn.Conv2d(32,32,3,padding=1, device = DEVICE),
                              nn.BatchNorm2d(32),
                nn.ReLU()
            ), in_dim=32, out_dim =32, stride = 1),
            self._wrap(
                nn.Sequential(nn.Conv2d(32,64,3,stride = 2, padding=1, device = DEVICE),
                              nn.BatchNorm2d(64),
                nn.ReLU()
            ), in_dim=32, out_dim = 64,stride = 2),
            self._wrap(
                nn.Sequential(nn.Conv2d(64,64,3,stride = 1, padding=1, device = DEVICE),
                              nn.BatchNorm2d(64),
                nn.ReLU()
            ), in_dim=64,out_dim = 64, stride = 1),
            self._wrap(
                nn.Sequential(nn.Conv2d(64, 128,3,stride = 2, padding=1, device = DEVICE),
                              nn.BatchNorm2d(128),
                nn.ReLU()
            ), in_dim=64, out_dim = 128,stride = 2),
            )
        self.exit = MHCExit() if mhc else nn.Identity()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    def _wrap(self, module, in_dim, out_dim,stride):
        if not self.mhc:
            return module
        return mHCWrapper(n_streams = self.n_streams, module = module, inpdim = in_dim, outdim=out_dim, stride=stride, useNS = self.useNS, static = self.static)
    def forward(self, x):
        x = self.stem(x)
        x = self.entry(x)
        x = self.backbone(x)
        x = self.exit(x)
        logits = self.head(x)
        return logits
        


