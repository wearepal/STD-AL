from configs.config import CelebAConfig

from .bit import ResNetV2


class CelebA(ResNetV2):
    def __init__(self, head_size, optim_cfg):
        super().__init__(optim_cfg, head_size)

    def with_embedding(self, x):
        embed = self.head[:3](self.body(self.root(x)))[..., 0, 0]
        out = self.head.conv(embed.view(-1, 2048, 1, 1))
        return embed, out[..., 0, 0]

    @property
    def fixed_embedding(self):
        return True


def get_model(cfg: CelebAConfig, optim_cfg):
    assert isinstance(cfg.sens_attr, str)
    return CelebA(2, optim_cfg)
