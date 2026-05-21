from mani_skill.utils.registration import register_env

from .two_robots_handover_active import TwoRobotsHandoverActiveEnv


@register_env("TwoRobotsHandoverActiveC-rf", max_episode_steps=500)
class TwoRobotsHandoverActiveCEnv(TwoRobotsHandoverActiveEnv):
    default_config_name = "two_robots_handover_active_C.yaml"
    forced_direction = None
