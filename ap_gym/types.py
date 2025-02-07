from typing import TypeVar, Dict, Literal, Union

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
PredType = TypeVar("PredType")
PredTargetType = TypeVar("PredTargetType")
ArrayType = TypeVar("ArrayType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")
WrapperPredType = TypeVar("WrapperPredType")
WrapperPredTargetType = TypeVar("WrapperPredTargetType")
WrapperArrayType = TypeVar("WrapperArrayType")

FullActType = Dict[Literal["action", "prediction"], Union[ActType, PredType]]
