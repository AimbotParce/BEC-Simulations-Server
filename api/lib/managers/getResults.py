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

    if not data["simulation_id"] in threadStatus:
        return {"message": "Simulation does not exist", "ok": False}

    if not threadStatus[data["simulation_id"]]["finished"]:
        return {"message": "Simulation not finished", "ok": False}

    folderName = threadStatus[data["simulation_id"]]["simulation_name"]

    # Get the results
    resultsFolder = os.path.join(constants.SIMULATIONS_FOLDER, folderName, "results")
    resultFile = None
    for file in os.listdir(resultsFolder):
        if file.endswith(".npy") and file.startswith(data["simulation_id"]):
            resultFile = os.path.join(resultsFolder, file)
            break

    if resultFile is None:
        return {"message": "No results found", "ok": False}

    results = np.load(resultFile, allow_pickle=True).item()
    print(results)

    return {"message": "Results", "ok": True, "results": results}
