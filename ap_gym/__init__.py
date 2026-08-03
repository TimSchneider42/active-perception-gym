try:
    from ._version import version as __version__
except ImportError:  # not installed / no build metadata available
    try:
        from importlib.metadata import version as _package_version

        __version__ = _package_version("ap_gym")
    except Exception:
        __version__ = "0.0.0"

from .util import idoc
from .active_classification_env import (
    ActiveClassificationEnv,
    ActiveClassificationVectorEnv,
    ActiveClassificationLogWrapper,
    ActiveClassificationVectorLogWrapper,
)
from .active_perception_env import (
    ActivePerceptionEnv,
    BaseActivePerceptionEnv,
    ActivePerceptionWrapper,
    ActivePerceptionRestoreWrapper,
    ActivePerceptionActionSpace,
    PseudoActivePerceptionWrapper,
    ensure_active_perception_env,
    NoActivePerceptionEnvError,
)
from .active_perception_vector_env import (
    ActivePerceptionVectorEnv,
    BaseActivePerceptionVectorEnv,
    ActivePerceptionVectorWrapper,
    ActivePerceptionVectorRestoreWrapper,
    PseudoActivePerceptionVectorWrapper,
    ensure_active_perception_vector_env,
)
from .active_regression_env import (
    ActiveRegressionEnv,
    ActiveRegressionVectorEnv,
    ActiveRegressionLogWrapper,
    ActiveRegressionVectorLogWrapper,
)
from .image_space import ImageSpace
from .loss_fn import LossFn, LambdaLossFn, ZeroLossFn, CrossEntropyLossFn, MSELossFn
from .time_limit import TimeLimit
from .vector_to_single_wrapper import (
    VectorToSingleWrapper,
    ActivePerceptionVectorToSingleWrapper,
)
from .logit_space import LogitSpace
from .sparsify_wrapper import SparsifyWrapper, SparsifyVectorWrapper

from .envs.registration import make, make_vec, register_envs, register

register_envs()
