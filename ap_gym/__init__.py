from .active_classification_env import (
    ActiveClassificationEnv,
    ActiveClassificationVectorEnv,
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
from .active_regression_env import ActiveRegressionEnv, ActiveRegressionVectorEnv
from .image_space import ImageSpace
from .loss_fn import LossFn, LambdaLossFn, ZeroLossFn, CrossEntropyLossFn, MSELossFn
from .time_limit import TimeLimit
from .vector_to_single_wrapper import (
    VectorToSingleWrapper,
    ActivePerceptionVectorToSingleWrapper,
)

from .envs.registration import make, make_vec, register_envs

register_envs()
