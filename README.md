# GeoMagPro — Geophysical Inversion Suite

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Python implementation of the PSO, ABIC, and Li-Oldenburg gravity and
aeromagnetic inversion suite developed for the PhD research programme at
Istanbul University, Institute of Marine Sciences and Management.
## [1.0.2] — 2026-03
Developed and applied to the **southern Marmara Sea** dataset as part of
a three-paper PhD publication strategy targeting *Marine Geophysical
Research* and *Int. J. Environ. Geoinformatics*.

---

## Companion publications
## [1.0.1] — 2025-09
> Aygün MA, Demirel AS *(2026)* Automated digitisation and
> integration of legacy geophysical maps: a reproducible Python workflow
> applied to the southern Marmara Sea.
> *Int. J. Environ. Geoinformatics*
## [1.0.2] — 2026-03
> Aygün MA, Demirel AS *(2026 in press-b)* Crustal density structure and
> basement configuration in the southern Marmara Sea: insights from
> integrated gravity analysis of legacy datasets.
> *Marine Geophysical Research*

> Aygün MA, Demirel AS *(2026 in press-c)* Magnetic basement architecture
> and fault system geometry in the southern Marmara Sea: multi-method
> aeromagnetic analysis across a continental transform.
> *Marine Geophysical Research*

---

## Overview

| Module | Method | Data type | Academic reference |
|--------|--------|-----------|--------------------|
| `geomagpro.pso_inversion` | Particle Swarm Optimisation | Gravity + Magnetic | Kennedy & Eberhart 1995; Pallero et al. 2015 |
| `geomagpro.abic_inversion` | Akaike Bayesian Information Criterion | Gravity + Magnetic | Akaike 1980; Murata 1993 |
| `geomagpro.li_oldenburg` | IRLS regularised inversion | Gravity + Magnetic | Li & Oldenburg 1996, 1998 |
| `geomagpro.grid_processing` | RAPS · TDR · THD · Upward continuation | Gravity + Magnetic | Spector & Grant 1970; Salem et al. 2007 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/edmadz/Inversion-Module.git
cd Inversion-Module

# Install dependencies
pip install -r requirements.txt

# Editable install (recommended for development)
pip install -e .
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.24, SciPy ≥ 1.11,
Matplotlib ≥ 3.7, Pandas ≥ 2.0.

---

## Quick start

```python
import numpy as np
from geomagpro import (
    PSOParams, run_pso_multi_profile,
    ABICParams, run_abic_gravity,
    LiOldenburgParams, run_li_oldenburg,
    raps_depth, tilt_derivative,
)

# Load your pre-processed grid (.npy with keys GS, lons, lats, dx)
data   = np.load('grids_v2.npy', allow_pickle=True).item()
GS     = data['GS']     # Bouguer anomaly (mGal)
lons   = data['lons']
lats   = data['lats']
dx_deg = data['dx']
dx_km  = dx_deg * 111.0

# RAPS depth estimation
raps = raps_depth(GS, dx_km=dx_km, n_segments=4)
print('Depth horizons (km):', raps['depths_km'])

# PSO inversion — 10 N-S profiles
pso_params = PSOParams(n_particles=25, n_iterations=100)
results = run_pso_multi_profile(
    GS, lons, lats, n_profiles=10,
    layer_tops_km=[0.52, 3.2, 8.0, 15.0],
    layer_dzs_km =[2.68, 4.8, 7.0, 10.0],
    dx_grid_km=dx_km, data_type='gravity',
    params=pso_params)
print(f'PSO RMS: {[f"{r.rms:.3f}" for r in results]}')

# ABIC 3D inversion
abic_result = run_abic_gravity(
    GS, lons, lats, dx_deg,
    depths_km=[0.52, 3.2, 8.0, 15.0],
    dzs_km   =[2.68, 4.8, 7.0, 10.0])
print(f'ABIC optimal ω = {abic_result.opt_omega:.4g}')
```

---

## Repository structure

```
geomagpro/
├── geomagpro/
│   ├── __init__.py          Public API exports
│   ├── pso_inversion.py     PSO gravity and magnetic inversion
│   ├── abic_inversion.py    ABIC 3D spectral inversion
│   ├── li_oldenburg.py      Li-Oldenburg IRLS inversion
│   └── grid_processing.py   RAPS, TDR, THD, upward continuation
├── examples/
│   ├── run_gravity_inversion.py   End-to-end gravity workflow
│   └── run_magnetic_inversion.py  End-to-end magnetic workflow
├── tests/
│   └── test_inversions.py   pytest unit tests (synthetic data)
├── requirements.txt
├── setup.py
└── README.md
```

---

## Running the examples

```bash
# Gravity (requires grids_v2.npy in working directory)
python examples/run_gravity_inversion.py --grid grids_v2.npy --out results_gravity

# Magnetic
python examples/run_magnetic_inversion.py --grid grids_v2.npy --out results_magnetic

# Skip Li-Oldenburg for a faster test run
python examples/run_gravity_inversion.py --skip-li --pso-iter 10
```

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

All tests use synthetic data and require no external files.

---

## Method descriptions

### PSO — Particle Swarm Optimisation

PSO minimises `||Km − d||²` over the model vector **m** bounded to
`[param_min, param_max]`.  Each particle updates its velocity
according to:

```
v_i(t+1) = w·v_i(t) + c₁·r₁·(p_best − x_i) + c₂·r₂·(g_best − x_i)
```

The gravity kernel is the 2D rectangular-prism formula of Telford et al.
(1990); the magnetic kernel is the Talwani-Heirtzler (1964) formulation
for RTP-transformed data.

**Key parameters:** `n_particles` (25–50), `inertia_w` (0.72),
`c1 = c2` (1.5), `param_min/max` (±0.8 g/cm³ for gravity; ±0.15 SI
for magnetic).

### ABIC — Akaike's Bayesian Information Criterion

ABIC selects the regularisation parameter ω by minimising:

```
ABIC(ω) = N·log σ²(ω) + 2·log₁₀(ω)
```

The spectral inversion is:

```
Δρ̂(k) = H*(k) · Ĝ_obs(k) / (|H(k)|² + ω·|k|⁴)
```

where `H(k) = 2πG · dz · exp(−2π|k|z)` for gravity and
`H(k) = G · dz · exp(−2π|k|z)` for magnetic (RTP).

**Key parameters:** `log_omega_min/max` (search range),
`n_grid_search` (30 is sufficient), `param_clip` (physical bounds).

### Li-Oldenburg IRLS

Minimises `φ(m) = ||Gm−d||² + λ(φ_s + φ_x)` using iteratively
reweighted depth-weighted norms.  Depth weighting:
`w(z) = 1/(z + ε)^(β/2)` with β = 1.5 (gravity) or β = 3.0
(magnetic).

**Key parameters:** `n_iterations` (100), `regularisation_lambda`
(0.3–0.5), `depth_weight_beta` (1.5 for gravity, 3.0 for magnetic).

---

## Physical constants and IGRF parameters (southern Marmara Sea)

| Parameter | Value |
|-----------|-------|
| Reference density | 2.67 g/cm³ |
| IGRF epoch | 1982.0 |
| Inclination | 57.2° |
| Declination | 3.8° |
| Total field | 47,850 nT |
| Grid cell size | 0.005° (~0.56 km) |
| Study area | 27.0–29.42°E, 40.17–40.76°N |

---

## Citation

If you use this code in your research, please cite the companion papers
listed above and this repository:

```bibtex
@misc{aygun2026geomagpro,
  author    = {Ayg\"{u}n, Muhammet Ali},
  title     = {{GeoMagPro}: {PSO}, {ABIC} and {Li-Oldenburg} geophysical
               inversion suite for gravity and aeromagnetic data},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/edmadz/Inversion-Module}
}
```

---

## References

Akaike H (1980) Likelihood and the Bayes procedure. In: Bayesian Statistics. Valencia, pp 143–166.

Essa KS, Elhussein M (2018) PSO for interpretation of magnetic anomalies. *Pure Appl Geophys* 175:3539–3553.

Kennedy J, Eberhart RC (1995) Particle swarm optimization. *Proc IEEE ICNN* 4:1942–1948.

Li Y, Oldenburg DW (1996) 3-D inversion of magnetic data. *Geophysics* 61:394–408.

Li Y, Oldenburg DW (1998) 3-D inversion of gravity data. *Geophysics* 63:109–119.

Miller HG, Singh V (1994) Potential field tilt. *J Appl Geophys* 32:213–217.

Murata Y (1993) Estimation of optimum average surficial density from gravity data. *J Geophys Res* 98:12097–12109.

Pallero JLG, Fernández-Martínez JL, Bonvalot S, Fudym O (2015) Gravity inversion via PSO. *J Appl Geophys* 116:180–191.

Salem A et al. (2007) Interpretation of magnetic data using tilt-angle derivatives. *Geophysics* 73:L1–L10.

Spector A, Grant FS (1970) Statistical models for interpreting aeromagnetic data. *Geophysics* 35:293–302.

Talwani M, Heirtzler JR (1964) Computation of magnetic anomalies. Stanford Univ Publ, pp 464–480.

Telford WM, Geldart LP, Sheriff RE (1990) *Applied Geophysics*, 2nd edn. Cambridge University Press.

Yabuki T, Matsu'ura M (1992) Geodetic data inversion using ABIC. *Geophys J Int* 109:363–375.

---

## Licence

MIT — see [LICENSE](LICENSE) for details.
