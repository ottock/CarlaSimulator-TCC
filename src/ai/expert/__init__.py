"""Expert policies that generate reference controls for behavioral cloning.

Stage A (CARLA Towns) uses ``basic_agent_expert.ExpertPolicy`` (wraps CARLA's
BasicAgent). Stage B (custom track) will add a Pure Pursuit expert. Both expose
the same call signature the model uses: ``policy(obs) -> (steer, throttle, brake)``.
"""
