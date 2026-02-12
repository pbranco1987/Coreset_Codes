"""
Brazil Telecom Loader -- Constants and small data structures.

Split out from ``brazil_telecom_loader.py`` for maintainability.
"""

from __future__ import annotations


# Brazilian state codes to numeric index (27 states + DF)
BRAZILIAN_STATES = [
    'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA',
    'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN',
    'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'
]
STATE_TO_IDX = {s: i for i, s in enumerate(BRAZILIAN_STATES)}

# Expected dataset dimensions per manuscript
EXPECTED_N = 5569  # Number of municipalities (Section 1)
EXPECTED_G = 27    # Number of states (Section 1)
