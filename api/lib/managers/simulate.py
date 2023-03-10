import json
import os
from importlib.machinery import SourceFileLoader
from types import ModuleType

from .. import constants
from ..BEC_simulations.lib import constants as BECconstants


def simulate(data):
    if not "name" in data:
        return {"message": "No simulation name given", "ok": False}

    folder = os.path.join(constants.SIMULATIONS_FOLDER, data["name"])
    if not os.path.exists(folder):
        return {"message": "Simulation does not exist", "ok": False}

    # Transform the cosntants to a dict
    simulationConstants = {}
    for constant in dir(BECconstants):
        if (
            not constant.startswith("__")
            and not callable(getattr(BECconstants, constant))
            and not isinstance(getattr(BECconstants, constant), ModuleType)
        ):
            simulationConstants[constant] = getattr(BECconstants, constant)

    # Overwrite constants with the simulation's constants (constants.py)
    fileConstants = SourceFileLoader("file_constants", os.path.join(folder, "constants.py")).load_module()
    for constant in dir(fileConstants):
        if (
            not constant.startswith("__")
            and not callable(getattr(fileConstants, constant))
            and not isinstance(getattr(fileConstants, constant), ModuleType)
        ):
            simulationConstants[constant] = getattr(fileConstants, constant)

    # Finally, overwrite the constants with the constants given in the request
    if "constants" in data:
        for constant in data["constants"]:
            simulationConstants[constant] = data["constants"][constant]

    print(simulationConstants)

    return {"message": "Simulation started", "ok": True}
