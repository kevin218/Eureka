"""Stage 5 model exports, sharing JAX implementations where supported."""
# Load the base classes before JAX models import them during initialization.
from .Model import Model, CompositeModel
from .AstroModel import AstroModel, PlanetParams
from .BatmanModels import BatmanTransitModel, BatmanEclipseModel
from .CatwomanModel import CatwomanTransitModel
from .FleckModel import FleckTransitModel
from .GPModel import GPModel
from .HarmonicaModel import HarmonicaTransitModel
from .KeplerOrbit import KeplerOrbit
from .PoetModels import PoetTransitModel, PoetEclipseModel, PoetPCModel

from ..jax_models import (
    CentroidModel,
    CommonModeModel,
    DampedOscillatorModel,
    ExpRampModel,
    HSTRampModel,
    LorentzianModel,
    PolynomialModel,
    QuasiLambertianPhaseCurve,
    SinusoidPhaseCurveModel,
    StepModel,
)
