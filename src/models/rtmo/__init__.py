"""
RTMOPose: Real-time multi-person pose estimation
Standalone PyTorch implementation inspired by RTMO (OpenMMLab, Apache 2.0)
Custom implementation - no MMPose/MMDet dependencies
"""

from .rtmo import RTMOPose
from .criterion import RTMOCriterion
from .postprocesses import RTMOPostProcess
