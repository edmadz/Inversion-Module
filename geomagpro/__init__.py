"""
GeoMagPro — Geophysical Inversion Suite
========================================
Gravity and aeromagnetic inversion methods developed for the
southern Marmara Sea PhD research programme.

Companion publication
---------------------
Aygün MA, Demirel AS (in press-b) Crustal density structure and
  basement configuration in the southern Marmara Sea: insights
  from integrated gravity analysis. Mar Geophys Res.

Aygün MA, Demirel AS (in press-c) Magnetic basement architecture
  and fault system geometry in the southern Marmara Sea:
  multi-method aeromagnetic analysis across a continental transform.
  Mar Geophys Res.

Modules
-------
pso_inversion    Particle Swarm Optimisation (PSO) gravity and
                 magnetic inversion along N-S profiles.
abic_inversion   ABIC 3D spectral inversion for gravity density
                 and magnetic susceptibility.
li_oldenburg     Li-Oldenburg IRLS 2D profile inversion.
grid_processing  RAPS, TDR, THD, upward continuation, lineament
                 extraction.

Quick start
-----------
>>> from geomagpro.pso_inversion import run_pso_multi_profile, PSOParams
>>> from geomagpro.abic_inversion import run_abic_gravity, ABICParams
>>> from geomagpro.li_oldenburg import run_li_oldenburg
>>> from geomagpro.grid_processing import tilt_derivative, raps_depth

License : MIT
Author  : Muhammet Ali Aygün, Istanbul University& Istanbul Tecnical University
Contact : maygun@ogr.iu.edu.tr
Version : 1.0.2
"""

__version__ = '1.0.2'
__author__  = 'Muhammet Ali Aygün'

from .pso_inversion  import (PSOParams, PSOResult,
                              run_pso_gravity_profile,
                              run_pso_magnetic_profile,
                              run_pso_multi_profile)

from .abic_inversion import (ABICParams, ABICResult,
                              run_abic_gravity,
                              run_abic_magnetic)

from .li_oldenburg   import (LiOldenburgParams, LiOldenburgResult,
                              run_li_oldenburg)

from .grid_processing import (raps_depth,
                               upward_continue,
                               tilt_derivative,
                               total_horizontal_derivative,
                               extract_tdr_lineaments)

__all__ = [
    'PSOParams', 'PSOResult',
    'run_pso_gravity_profile', 'run_pso_magnetic_profile',
    'run_pso_multi_profile',
    'ABICParams', 'ABICResult',
    'run_abic_gravity', 'run_abic_magnetic',
    'LiOldenburgParams', 'LiOldenburgResult',
    'run_li_oldenburg',
    'raps_depth', 'upward_continue',
    'tilt_derivative', 'total_horizontal_derivative',
    'extract_tdr_lineaments',
]
