from __future__ import annotations

import math
from collections import deque


class HydVolCap:
    def __init__(
        self,
        vol_low_thr: float,
        vol_high_thr: float,
        cap_low_vol: int = 200,
        cap_high_vol: int = 50,
        window: int = 200,
    ) -> None:
        self.vol_low_thr = float(vol_low_thr)
        self.vol_high_thr = float(vol_high_thr)
        self.cap_low_vol = int(cap_low_vol)
        self.cap_high_vol = int(cap_high_vol)
        self.window = int(window)
        self.prices: deque[float] = deque(maxlen=self.window)
        self._cap = self.cap_low_vol
        self._stdev = 0.0

    def update(self, mid_price: float) -> None:
        self.prices.append(float(mid_price))
        if len(self.prices) < self.window:
            self._stdev = 0.0
            self._cap = self.cap_low_vol
            return

        mean = sum(self.prices) / len(self.prices)
        variance = sum((price - mean) ** 2 for price in self.prices) / len(self.prices)
        self._stdev = math.sqrt(max(variance, 0.0))

        if self._stdev <= self.vol_low_thr:
            self._cap = self.cap_low_vol
        elif self._stdev >= self.vol_high_thr:
            self._cap = self.cap_high_vol
        elif self.vol_high_thr <= self.vol_low_thr:
            self._cap = self.cap_high_vol
        else:
            ratio = (self._stdev - self.vol_low_thr) / (self.vol_high_thr - self.vol_low_thr)
            interpolated = self.cap_low_vol + ratio * (self.cap_high_vol - self.cap_low_vol)
            self._cap = int(round(interpolated))

    def get_cap(self) -> int:
        return self._cap
