from mani_skill.utils.registration import register_env

from .two_robots_handover_active import TwoRobotsHandoverActiveEnv


@register_env("TwoRobotsHandoverActiveA-rf", max_episode_steps=500)
class TwoRobotsHandoverActiveAEnv(TwoRobotsHandoverActiveEnv):
    default_config_name = "two_robots_handover_active_A.yaml"
    forced_direction = "left_to_right"
