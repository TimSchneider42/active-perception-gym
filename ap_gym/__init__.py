from .active_classification_env import (
    ActiveClassificationEnv,
    ActiveClassificationVectorEnv,
    CrossEntropyLossFn,
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
from .circle_square_env import CircleSquareEnv, CircleSquareVectorEnv
from .image_perception_vector_env import ImagePerceptionVectorEnv
from .image_space import ImageSpace
from .loss_fn import LossFn, LambdaLossFn, ZeroLossFn
from .mnist_env import MNISTEnv, MNISTVectorEnv
from .registration import make, make_vec, register_envs
from .vector_to_single_wrapper import (
    VectorToSingleWrapper,
    ActivePerceptionVectorToSingleWrapper,
)

register_envs()
