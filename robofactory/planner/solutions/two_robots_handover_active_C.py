from robofactory.tasks import TwoRobotsHandoverActiveCEnv
from robofactory.planner.solutions.two_robots_handover_active_A import solve as solve_a
from robofactory.planner.solutions.two_robots_handover_active_B import solve as solve_b


def solve(env: TwoRobotsHandoverActiveCEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    if int(env.unwrapped.source_agent_id) == 0:
        return solve_a(env, seed=None, debug=debug, vis=vis, reset=False)
    return solve_b(env, seed=None, debug=debug, vis=vis, reset=False)
