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
    ensure_active_perception_vector_env,
)
from .active_classification_env import (
    ActiveClassificationEnv,
    ActiveClassificationVectorEnv,
    CrossEntropyLossFn,
)
from .vector_to_single_wrapper import (
    VectorToSingleWrapper,
    ActivePerceptionVectorToSingleWrapper,
)
from .image_space import ImageSpace
from .image_perception_vector_env import ImagePerceptionVectorEnv
from .mnist_env import MNISTEnv, MNISTVectorEnv
from .circle_square_env import CircleSquareEnv, CircleSquareVectorEnv
from .loss_fn import LossFn, LambdaLossFn, ZeroLossFn
from .registration import make, make_vec, register_envs

register_envs()
