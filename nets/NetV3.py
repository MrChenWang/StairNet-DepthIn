from nets.modules import *


class StairNet_DepthIn(nn.Module):
    def __init__(self, width=1.0):
        super(StairNet_DepthIn, self).__init__()
        # Focus module
        self.initial = Focus(3, 64, width=width)
        # Backbone of SE-ResNeXt blocks
        self.resblock1 = nn.Sequential(
            ResBlockX_SE(64, 256, stride=2, width=width),
            ResBlockX_SE(256, 256, width=width),
            ResBlockX_SE(256, 256, width=width)
        )
        self.resblock2 = nn.Sequential(
            ResBlockX_SE(256, 512, stride=2, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=2, dilation=2, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=4, dilation=4, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=8, dilation=8, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=16, dilation=16, width=width),
        )
        self.resblock3 = nn.Sequential(
            ResBlockX_SE(512, 512, stride=2, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=2, dilation=2, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=4, dilation=4, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=8, dilation=8, width=width),
            ResBlockX_SE(512, 512, width=width),
            ResBlockX_SE_Super1_1(512, 512, padding=16, dilation=16, width=width),
        )
        # Depth input branch
        self.branch2 = nn.Sequential(
            Focus(1, 64, width=width),
            ResBlockX_SE(64, 256, stride=2, width=width),
            ResBlockX_SE(256, 256, width=width),
            ResBlockX_SE(256, 256, width=width)
        )
        self.semoudle = SEmoudle(256, 256, width=width)
        # Necks
        self.neck = nn.Sequential(
            nn.Conv2d(int(512 * width), int(256 * width), 3, stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(int(256 * width)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(int(256 * width), int(256 * width), kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(int(256 * width)),
            nn.ReLU(inplace=True)
        )
        self.neck_b = nn.Sequential(
            nn.Conv2d(int(256 * width), int(128 * width), 3, stride=1, padding=1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True)
        )
        self.neck_r = nn.Sequential(
            nn.Conv2d(int(256 * width), int(128 * width), 3, stride=1, padding=1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True)
        )
        # Output branches
        self.conf_branch_b = nn.Sequential(
            nn.Conv2d(int(128 * width), int(128 * width), 3, 1, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 * width), 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.loc_branch_b = nn.Sequential(
            nn.Conv2d(int(128 * width), int(128 * width), 3, 1, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 * width), 4, 1, 1, 0),
            nn.Sigmoid()
        )
        self.conf_branch_r = nn.Sequential(
            nn.Conv2d(int(128 * width), int(128 * width), 3, 1, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 * width), 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.loc_branch_r = nn.Sequential(
            nn.Conv2d(int(128 * width), int(128 * width), 3, 1, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 * width), 4, 1, 1, 0),
            nn.Sigmoid()
        )
        # Segmentation output branch with Up-sampling
        self.mask_branch = nn.Sequential(
            nn.Conv2d(int(512 * width), int(256 * width), 3, 1, 1),
            nn.BatchNorm2d(int(256 * width)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(int(256 * width), int(256 * width), 4, 2, 1),
            nn.BatchNorm2d(int(256 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(256 * width), int(128 * width), 3, 1, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(int(128 * width), int(128 * width), 4, 2, 1),
            nn.BatchNorm2d(int(128 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 * width), int(64 * width), 3, 1, 1),
            nn.BatchNorm2d(int(64 * width)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(int(64 * width), int(64 * width), 4, 2, 1),
            nn.BatchNorm2d(int(64 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(64 * width), int(32 * width), 3, 1, 1),
            nn.BatchNorm2d(int(32 * width)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(int(32 * width), int(32 * width), 4, 2, 1),
            nn.BatchNorm2d(int(32 * width)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(32 * width), 3, 1, 1, 0)
        )

    def forward(self, x1, x2):
        x1 = self.initial(x1)
        x1 = self.resblock1(x1)
        x2 = self.branch2(x2)
        x = self.semoudle(x1, x2)
        x = self.resblock2(x)
        x = self.resblock3(x)
        f3 = self.mask_branch(x)
        x = self.neck(x)
        f_b = self.neck_b(x)
        f_r = self.neck_r(x)
        fb1 = self.conf_branch_b(f_b)
        fb2 = self.loc_branch_b(f_b)
        fr1 = self.conf_branch_r(f_r)
        fr2 = self.loc_branch_r(f_r)
        return fb1, fb2, fr1, fr2, f3


if __name__ == "__main__":
    x1 = torch.randn(1, 3, 512, 512)
    x2 = torch.randn(1, 1, 512, 512)
    stairnet = StairNet_DepthIn(width=1.0)
    stairnet.eval()
    # stat(resnet, (3, 512, 512))
    y = stairnet(x1, x2)
    y1, y2, y3, y4, y5 = y
    print(y1.size(), y2.size(), y3.size(), y4.size(), y5.size())
