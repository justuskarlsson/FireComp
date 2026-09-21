"""
firecomp.next_day — Next-day fire spread prediction (per-pixel segmentation).

Layout:
    config.py               - NextDayConfig dataclass
    dataset.py              - NextDayDataset, Batch, Sample, field declarations
    preprocess.py           - pick / download / build CLI pipeline
    implementations/
        dl_2d.py            - 2D UNet segmentation, train + eval
        grid_search.py      - model x loss x target benchmark grid
        transfer.py         - cross-region transfer matrix / leave-one-out
        ablation.py         - input feature-group ablation
        fire_type_study.py  - fire type composition experiments
        persistence.py, morphological.py - baselines
"""
