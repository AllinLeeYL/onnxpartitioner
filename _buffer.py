from dataclasses import dataclass

@dataclass
class Buffer:
    channel_s: int
    pixel_s: int

    def __gt__(self, other):
        return self.channel_s > other.channel_s or self.pixel_s > other.pixel_s