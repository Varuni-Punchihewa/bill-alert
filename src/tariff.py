"""Translate units into an estimated bill (LKR) with the PUCSL domestic tariff.

Used by the app and the report to show *why* the 60-unit line matters. It is
an estimate: it ignores taxes/levies and assumes a 30-day billing period
(block sizes are scaled for other lengths, as the tariff specifies).
"""
from src.config import BILLING_BASE_DAYS, TARIFF_HIGH, TARIFF_LOW, TARIFF_VERY_HIGH


def _blocks_cost(units, blocks, scale):
    cost, lower, fixed = 0.0, 0.0, blocks[0][2]
    for upper, rate, fix in blocks:
        upper = upper * scale
        if units > lower:
            cost += (min(units, upper) - lower) * rate
            fixed = fix
        lower = upper
    return cost + fixed


def estimate_bill(units: float, days: int = BILLING_BASE_DAYS) -> float:
    scale = days / BILLING_BASE_DAYS
    if units <= 60 * scale:
        return round(_blocks_cost(units, TARIFF_LOW, scale), 2)
    if units <= 180 * scale:
        return round(_blocks_cost(units, TARIFF_HIGH, scale), 2)
    return round(_blocks_cost(units, TARIFF_VERY_HIGH, scale), 2)
