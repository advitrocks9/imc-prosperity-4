"""Conversion arbitrage (Archetype 5: MACARONS / ORCHIDS).

Sell locally above import cost, convert the short position each tick via
the external market. Never hold long inventory (storage costs bleed).

Parameters to calibrate: min_edge (0.2 default, lower captures more edge), conv_limit (10).
"""
import math
from datamodel import Order, OrderDepth, ConversionObservation
from strategies._base import conversion_arb


class ConversionTrader:
    """Cross-exchange conversion arbitrage.

    Args:
        product: product symbol
        limit: position limit
        conv_limit: max conversions per tick
        min_edge: minimum profit above import cost to trade
    """

    def __init__(
        self,
        product: str,
        limit: int = 75,
        conv_limit: int = 10,
        min_edge: float = 0.2,
    ) -> None:
        self.product = product
        self.limit = limit
        self.conv_limit = conv_limit
        self.min_edge = min_edge

    def run(
        self,
        od: OrderDepth,
        obs: ConversionObservation,
        pos: int,
    ) -> tuple[list[Order], int]:
        """Generate orders and conversion request.

        Returns:
            (orders, conversions_requested)
        """
        return conversion_arb(
            product=self.product,
            obs=obs,
            local_od=od,
            pos=pos,
            limit=self.limit,
            conv_limit=self.conv_limit,
            min_edge=self.min_edge,
        )
