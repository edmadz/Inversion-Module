# Changelog

## [1.0.0] — 2024-03

### Added
- `geomagpro.pso_inversion` — PSO 2D gravity and magnetic profile inversion
  (Kennedy & Eberhart 1995; Pallero et al. 2015; Essa & Elhussein 2018)
- `geomagpro.abic_inversion` — ABIC 3D spectral inversion for gravity and
  magnetic data with automatic regularisation selection
  (Akaike 1980; Murata 1993; Yabuki & Matsu'ura 1992)
- `geomagpro.li_oldenburg` — Li-Oldenburg IRLS 2D profile inversion with
  depth weighting for gravity (β=1.5) and magnetic (β=3.0) data
  (Li & Oldenburg 1996, 1998)
- `geomagpro.grid_processing` — RAPS depth estimation, upward continuation,
  TDR, THD, lineament extraction (Spector & Grant 1970; Salem et al. 2007)
- End-to-end example scripts for gravity and magnetic workflows
- Full pytest test suite using synthetic data (no external files required)
- GitHub Actions CI for Python 3.9, 3.10, 3.11

### Applied to  — 2024-03
- Southern Marmara Sea gravity dataset: 49,020 Bouguer observations,
  86 lines, UTM Zone 35N
- Southern Marmara Sea aeromagnetic dataset: 25,230 RTP observations,
  87 lines, IGRF 1982.0
- Results published in Aygün and Demirel (in press-b, in press-c),
  Marine Geophysical Research
