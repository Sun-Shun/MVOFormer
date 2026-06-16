import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if downsample is not None:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes)
        else:
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.weight.requires_grad = True
                m.bias.requires_grad = True

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class TemporalEncoder(nn.Module):
    def __init__(self, num_bev_queue: int, embed_dims: int, num_block: int):
        super().__init__()

        self.num_bev_queue = num_bev_queue
        self.embed_dims = embed_dims
        in_channels = num_bev_queue * embed_dims
        out_channels = embed_dims

        temporal_block = [Bottleneck(in_channels, in_channels // 4) for _ in range(num_block - 1)]
        temporal_block.append(Bottleneck(
            in_channels,
            out_channels,
            downsample=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        ))
        self.temporal_block = nn.Sequential(*temporal_block)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                module.weight.requires_grad = True
                module.bias.requires_grad = True

    def forward(self, bev_feat: torch.Tensor, prev_bev: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            bev_feat: (bs, embed_dims, H, W) current BEV features.
            prev_bev: (bs, embed_dims, H, W) previous BEV features.
        Returns:
            (bs, H*W, embed_dims) temporally fused BEV features.
        """
        assert bev_feat.shape[1] == self.embed_dims, "BEV feature dims mismatch!"
        bs, dim, h, w = bev_feat.shape
        if prev_bev is None:
            prev_bev = bev_feat

        bev_queue = torch.cat([prev_bev, bev_feat], dim=1)
        temporal_fused_bev_feat = self.temporal_block(bev_queue)
        temporal_fused_bev_feat = temporal_fused_bev_feat.reshape(
            temporal_fused_bev_feat.shape[0], self.embed_dims, -1)
        temporal_fused_bev_feat = temporal_fused_bev_feat.permute(0, 2, 1)
        return temporal_fused_bev_feat
