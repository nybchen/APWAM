from robofactory.tasks import TwoRobotsHandoverActiveCEnv
from robofactory.planner.solutions.two_robots_handover_active import solve_handover


def solve(env: TwoRobotsHandoverActiveCEnv, seed=None, debug=False, vis=False):
    """Mixed handover: the environment samples left-to-right or right-to-left."""
    return solve_handover(env, seed=seed, debug=debug, vis=vis)
