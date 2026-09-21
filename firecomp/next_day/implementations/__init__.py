"""
firecomp.next_day.implementations — Concrete training scripts for next_day task.

Each file is a complete train + eval script with a CLI. No shared base class —
just a ~80 line story per implementation.

Available implementations:
    dl_2d.py            2D UNet segmentation (train + eval)
    grid_search.py      36-config Cartesian grid search (Paper 1 benchmark)
    transfer.py         Cross-region transfer matrix + leave-one-out (Paper 1)
"""
