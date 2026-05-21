from mani_skill.utils.registration import register_env

from .two_robots_handover_active import TwoRobotsHandoverActiveEnv


@register_env("TwoRobotsHandoverActiveB-rf", max_episode_steps=500)
class TwoRobotsHandoverActiveBEnv(TwoRobotsHandoverActiveEnv):
    default_config_name = "two_robots_handover_active_B.yaml"
    forced_direction = "right_to_left"
