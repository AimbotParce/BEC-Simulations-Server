import base64
import io
import os

import numpy as np

from .. import constants
from .simulate import threadStatus


def get_result(data):
    """
    Get the results of the simulation
    """
    if not "simulation_id" in data:
        return {"message": "No simulation id given", "ok": False}

    resultFile = None
    for root, dirs, files in os.walk(constants.SIMULATIONS_FOLDER):
        for file in files:
            if file.endswith(".npy") and file.startswith(data["simulation_id"]):
                resultFile = os.path.join(root, file)
                break

    if resultFile is None:
        return {"message": "No results found", "ok": False}

    matrix = np.load(resultFile, allow_pickle=True)

    buffer = io.BytesIO()

    np.save(buffer, matrix)

    result = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # To decode: np.load(io.BytesIO(base64.b64decode(response["result"])), allow_pickle=True)

    return {"message": "Results", "ok": True, "result": result}
