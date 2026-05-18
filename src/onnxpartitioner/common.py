from dataclasses import dataclass

@dataclass
class Buffer:
    channel_s: int
    pixel_s: int

    def __gt__(self, other):
        return self.channel_s > other.channel_s or self.pixel_s > other.pixel_s


@dataclass
class ConvSpec:
    # Input
    in_h: int
    in_w: int
    in_name: str
    in_channel: int

    # Output
    out_h: int
    out_w: int
    out_name: str
    out_channel: int

    # kernel
    k_h: int
    k_w: int
    k_name: str

    # bias
    b_name: str

    # other params
    pads: tuple
    strides: tuple
    batch: int


def max_multiplier_within_limit(base: int, limit: int) -> int:
    return 0 if base == 0 else limit // base