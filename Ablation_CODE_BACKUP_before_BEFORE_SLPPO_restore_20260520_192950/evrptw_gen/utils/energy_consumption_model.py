import numpy as np

def consumption_model(env, model_type = None, *args, **kwargs):
    if not model_type:
        return env['consumption_per_distance']
    else:
        raise NotImplementedError(f"Have not impleted energy model : {type}")
