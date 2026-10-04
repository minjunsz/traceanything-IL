"""Per-subtask finite state machines (strategies) and their registry."""

from markovian_policy.experts.fsm.base import GripperCmd, SubtaskFSM, at_home
from markovian_policy.experts.fsm.bottomknob import BottomknobFSM
from markovian_policy.experts.fsm.hinge import HingeFSM
from markovian_policy.experts.fsm.kettle import KettleFSM
from markovian_policy.experts.fsm.light import LightFSM
from markovian_policy.experts.fsm.microwave import MicrowaveFSM
from markovian_policy.experts.fsm.slide import SlideFSM
from markovian_policy.experts.fsm.topknob import TopknobFSM

SUBTASK_FSMS: dict[str, SubtaskFSM] = {
    "microwave": MicrowaveFSM(),
    "kettle": KettleFSM(),
    "slide": SlideFSM(),
    "light": LightFSM(),
    "topknob": TopknobFSM(),
    "bottomknob": BottomknobFSM(),
    "hinge": HingeFSM(),
}

__all__ = ["SUBTASK_FSMS", "GripperCmd", "SubtaskFSM", "at_home"]
