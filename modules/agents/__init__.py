"""
NeuralTrade Agents Module
Author: Erdinc Erdogan
"""

from .bear import *
from .bull import *
from .cross_agent_memory import *
from .deadlock_recovery import *
from .judge import *
from .judge_hardened import *
from .priors_engine import *
from .sanitized_memory import *
from .swarm import *
from .swarm_hardened import *

__all__ = [
    'BearAgent', 'BullAgent', 'CrossAgentMemory', 'DeadlockRecovery',
    'Judge', 'JudgeHardened', 'PriorsEngine', 'SanitizedMemory',
    'Swarm', 'SwarmHardened',
]