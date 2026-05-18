"""Static competitor / partner ticker lists for rich multi-name feature engineering.

These maps are **not** from a live LLM: they are fixed, reproducible pairs for coursework
and dataset storytelling (e.g. Apple ↔ TSMC supply chain). Use Yahoo Finance symbols.

Unknown tickers resolve to empty lists; the pipeline still runs with target + benchmark only.
"""

from __future__ import annotations

# Major peers / rivals (illustrative, editable for your report).
COMPETITOR_MAP: dict[str, list[str]] = {
    "AAPL": ["MSFT", "GOOGL", "DELL", "HPQ"],
    "MSFT": ["AAPL", "GOOGL", "ORCL", "CRM"],
    "GOOGL": ["META", "MSFT", "AAPL", "SNAP"],
    "META": ["GOOGL", "SNAP", "PINS", "NFLX"],
    "NVDA": ["AMD", "INTC", "AVGO", "QCOM"],
    "AMD": ["NVDA", "INTC", "QCOM", "AVGO"],
    "AMZN": ["WMT", "TGT", "EBAY", "SHOP"],
    "TSLA": ["F", "GM", "RIVN", "LCID"],
    "INTC": ["AMD", "NVDA", "QCOM", "AVGO"],
    "AVGO": ["QCOM", "MRVL", "TXN", "ADI"],
    "TSM": ["INTC", "AMD", "UMC", "QCOM"],
    "QCOM": ["AVGO", "MRVL", "SWKS", "QRVO"],
}

# Suppliers / ecosystem names (TSMC for Apple, foundries for semis, etc.).
PARTNER_MAP: dict[str, list[str]] = {
    "AAPL": ["TSM", "QCOM", "AVGO", "SWKS"],
    "MSFT": ["NVDA", "AMD", "INTC", "ORCL"],
    "GOOGL": ["NVDA", "AMD", "AVGO", "EQIX"],
    "META": ["SNAP", "PINS", "EQIX", "DLR"],
    "NVDA": ["TSM", "AVGO", "AMD", "MRVL"],
    "AMD": ["TSM", "NVDA", "QCOM", "UMC"],
    "AMZN": ["UPS", "FDX", "WMT", "CHWY"],
    "TSLA": ["PANW", "NXPI", "ON", "STM"],
    "INTC": ["TSM", "UMC", "AMAT", "LRCX"],
    "AVGO": ["TSM", "QCOM", "MRVL", "SWKS"],
    "TSM": ["ASX", "UMC", "AMAT", "LRCX"],
    "QCOM": ["TSM", "AVGO", "SWKS", "QRVO"],
}


def peers_for_ticker(ticker: str) -> tuple[list[str], list[str]]:
    """Return deduplicated (competitors, partners) for ``ticker`` (case-insensitive)."""
    key = ticker.strip().upper()
    comp = list(dict.fromkeys(COMPETITOR_MAP.get(key, [])))
    part = list(dict.fromkeys(PARTNER_MAP.get(key, [])))
    # Never include the target itself if maps overlap.
    comp = [c for c in comp if c.upper() != key]
    part = [p for p in part if p.upper() != key and p.upper() not in {c.upper() for c in comp}]
    return comp, part
