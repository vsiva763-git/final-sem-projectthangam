"""
Speech Enhancement Package
"""

from .model import SpeechEnhancementNetwork
from .components import (
    UnifiedDiscoveryUnit,
    TemporalSpectralCrossAttention,
    TwinSegmentAttentionIntegration,
    GaussianWeightedProgressiveStructure,
)
from .losses import CombinedLoss, ComplexValueLoss, MagnitudeLoss
from .data_processing import STFTProcessor, PowerLawCompression
from .trainer import Trainer, ModelCheckpoint, TrainingMonitor
from .dataset import SpeechEnhancementDataset, SyntheticNoiseDataset

__all__ = [
    'SpeechEnhancementNetwork',
    'UnifiedDiscoveryUnit',
    'TemporalSpectralCrossAttention',
    'TwinSegmentAttentionIntegration',
    'GaussianWeightedProgressiveStructure',
    'CombinedLoss',
    'ComplexValueLoss',
    'MagnitudeLoss',
    'STFTProcessor',
    'PowerLawCompression',
    'Trainer',
    'ModelCheckpoint',
    'TrainingMonitor',
    'SpeechEnhancementDataset',
    'SyntheticNoiseDataset',
]
