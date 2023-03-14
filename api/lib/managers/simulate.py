import argparse
import inspect
import json
import logging
import os
import threading
import uuid
from importlib.machinery import SourceFileLoader
from types import ModuleType

import jax
import jax.numpy as jnp

from .. import constants
from ..BEC_simulations.lib import constants as BECconstants
from ..BEC_simulations.lib.managers.crankNicolson import default as crankNicolson
from ..BEC_simulations.run import getSimulatorModule
from ..BEC_simulations.run import run as BECrun

jax.config.update("jax_enable_x64", True)
threadStatus = {}

logging.getLogger("BECsimulations").setLevel(logging.CRITICAL)


def loadWaveFunctionAndPotential(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")

    if not path.endswith(".py"):
        raise ValueError(f"File {path} must be a Python file")

    module = SourceFileLoader("module", path).load_module()
    if not hasattr(module, "waveFunction"):
        raise AttributeError(f"File {path} must have a waveFunction function")
    if not inspect.isfunction(module.waveFunction):
        raise AttributeError(f"waveFunction must be a function")
    waveFunctionGenerator = module.waveFunction
    # Check if function has the right signature
    signature = list(inspect.signature(waveFunctionGenerator).parameters.keys())
    if not signature == ["x", "t"]:
        raise AttributeError(
            f"waveFunction must have the signature waveFunction(x, t), but has waveFunction({', '.join(signature)})"
        )

    if not hasattr(module, "V"):
        raise AttributeError(f"File {path} must have a potential function (V)")
    if not inspect.isfunction(module.V):
        raise AttributeError(f"V must be a function")
    V = module.V
    # Check if function has the right signature
    signature = list(inspect.signature(V).parameters.keys())
    if not signature == ["x", "t"]:
        raise AttributeError(f"V must have the signature V(x, t), but has V({', '.join(signature)})")

    return waveFunctionGenerator, V


def simulate(data):
    if not "simulation_name" in data:
        return {"message": "No simulation name given", "ok": False}

    folder = os.path.join(constants.SIMULATIONS_FOLDER, data["simulation_name"])
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

    simulationID = str(uuid.uuid4())
    while simulationID in threadStatus:
        simulationID = str(uuid.uuid4())

    # Simulation metadata
    with open(os.path.join(folder, "simulation.json"), "r") as f:
        simulationMetadata = json.load(f)

    # Arguments
    args = argparse.Namespace()
    args.input = os.path.join(folder, "simulation.py")

    # Get the simulator
    CNModule = getSimulatorModule(simulationMetadata.get("crank_nicolson_simulator", None))

    # Run the simulation in a new thread
    threadStatus[simulationID] = {
        "percent": 0,
        "finished": False,
        "status": "not running",
        "simulation_id": simulationID,
        "simulation_name": data["simulation_name"],
        "error": None,
    }
    thread = threading.Thread(
        name=simulationID,
        target=__runSimulation,
        args=(args, simulationConstants, CNModule, threadStatus[simulationID]),
    )

    thread.start()

    return {"message": "Simulation started", "ok": True, "simulation_id": simulationID}


def __runSimulation(arguments, simConstants, CNModule, threadStatus):
    try:
        # Run the simulation
        threadStatus["status"] = "running"

        BECrun(arguments, simConstants, CNModule, threadStatus)

        threadStatus["finished"] = True
        threadStatus["percent"] = 100
        threadStatus["status"] = "finished"
    except Exception as e:
        threadStatus["error"] = str(e)
        threadStatus["status"] = "error"
        raise e
