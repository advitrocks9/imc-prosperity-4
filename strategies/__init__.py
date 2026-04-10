# strategies/ - Modular strategy code, inlined by build.py into submission
#
# Each module is independently importable. The build script resolves
# dependencies and concatenates them into a single file.
#
# _base.py    - shared components (wall_mid, EMA, BS suite, etc.)
# stable.py   - stable product MM (Archetype 1)
# drifting.py - drifting product MM (Archetype 2)
# basket.py   - basket/ETF arb (Archetype 3)
# options.py  - options IV scalping (Archetype 4)
# conversion.py - conversion arb (Archetype 5)
# signal.py   - signal-driven (Archetype 6)
