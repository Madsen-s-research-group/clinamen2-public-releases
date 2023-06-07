"""JSONEncoder that handles DeviceArrays.

    Removed from file_handling to avoid jnp dependency.
"""
import json
from collections import deque

import jax.numpy as jnp
import numpy as np

# If use case arises, add decoder and keep DeviceArray datatype.


class JSONEncoderwithJNP(json.JSONEncoder):
    """Class that extends JSONEncoder to handle different data types.

    Separate version that encodes JNP arrays to numpy.
    """

    def default(self, o):
        """Return a json-izable version of o or delegate on the base class."""
        if isinstance(o, np.generic):
            # Deal with non-serializable types such as numpy.int64
            return o.item()
        elif isinstance(o, np.ndarray):
            nruter = {
                "main_type": "NumPy/" + o.dtype.name,
                "data": o.tolist(),
            }
            return nruter
        elif isinstance(o, jnp.DeviceArray):
            nruter = {
                "main_type": "NumPy/" + o.dtype.name,
                "data": o.tolist(),
            }
            return nruter
        elif isinstance(o, deque):
            nruter = {
                "main_type": "deque/" + str(o.maxlen),
                "data": list(o),
            }
            return nruter
        return json.JSONEncoder.default(self, o)
