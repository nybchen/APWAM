from robofactory.tasks import TwoRobotsHandoverActiveBEnv
from robofactory.planner.solutions.two_robots_handover_active import solve_handover


def solve(env: TwoRobotsHandoverActiveBEnv, seed=None, debug=False, vis=False):
    """Right-to-left handover: agent 1 gives the cube to agent 0."""
    return solve_handover(
        env, seed=seed, debug=debug, vis=vis, source_id=1, target_id=0
    )
