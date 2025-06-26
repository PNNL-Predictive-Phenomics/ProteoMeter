from __future__ import annotations

from proteometer.lip_analysis import lip_analysis
from proteometer.params import Params
from proteometer.ptm_analysis import ptm_analysis

drop_samples = [
    "Infected_24h_1",
    "Mock_24h_1",
]
drop_samples = None
lip_params = Params("demo_data/lip.toml")
lip_pept, lip_site, lip_prot = lip_analysis(lip_params, drop_samples=drop_samples)
lip_pept.to_csv("demo_data/lip_processed_pept.csv")
lip_site.to_csv("demo_data/lip_processed_site.csv")
lip_prot.to_csv("demo_data/lip_processed_prot.csv")

ptm_params = Params("demo_data/ptm.toml")
ptm_site, ptm_prot = ptm_analysis(ptm_params)
ptm_site.to_csv("demo_data/ptm_processed_site.csv")
ptm_prot.to_csv("demo_data/ptm_processed_prot.csv")
