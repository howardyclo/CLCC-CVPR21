import math
import numpy as np
from collections import OrderedDict

def angular_error(estimation, ground_truth):
    return math.acos(np.clip(np.dot(estimation, ground_truth) / np.linalg.norm(estimation) / np.linalg.norm(ground_truth), -1, 1)) * 180 / math.pi

def percentiles(vals):
    vals = sorted(vals)

    def g(f):
        return np.percentile(vals, f * 100)

    median = g(0.5)
    mean = np.mean(vals)
    trimean = 0.25 * (g(0.25) + 2 * g(0.5) + g(0.75))
    
    results = {
      '25': np.mean(vals[:int(0.25 * len(vals))]),
      '75': np.mean(vals[int(0.75 * len(vals)):]),
      '95': g(0.95),
      'tri': trimean,
      'med': median,
      'mean': mean
    }

    return results

def valuesdict_to_percentilesdict(valdict):
    """ Convert values in dictionary to percentiles. """
    percentiledict = OrderedDict()
    for k, v in valdict.items():
        try:
            percentiledict.update([(k, percentiles(v))])
        except:
            print('WARNING:Ignore computing "%s" percentiles' % k)

    return percentiledict