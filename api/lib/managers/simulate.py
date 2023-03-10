import inspect
import json
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

jax.config.update("jax_enable_x64", True)


threadStatus = {}


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

    waveFunctionGenerator, V = loadWaveFunctionAndPotential(os.path.join(folder, "simulation.py"))

    simulationID = str(uuid.uuid4())
    while simulationID in threadStatus:
        simulationID = str(uuid.uuid4())

    # Run the simulation in a new thread
    threadStatus[simulationID] = {
        "percent": 0,
        "finished": False,
        "status": "not running",
        "simulation_id": simulationID,
        "name": data["name"],
    }
    thread = threading.Thread(
        name=simulationID,
        target=__runSimulation,
        args=(simulationConstants, waveFunctionGenerator, V, threadStatus[simulationID]),
    )

    thread.start()

    return {"message": "Simulation started", "ok": True, "simulation_id": simulationID}


def __runSimulation(simConstants, waveFunctionGenerator, V, threadStatus):
    # Run the simulation
    threadStatus["status"] = "running"
    x = jnp.arange(simConstants["xMin"], simConstants["xMax"], simConstants["dx"])
    t = jnp.arange(simConstants["tMin"], simConstants["tMax"], simConstants["dt"])

    waveFunctionGenerator = jax.jit(waveFunctionGenerator)
    V = jax.jit(V)

    psi = jnp.zeros((len(t), len(x)), dtype=jnp.complex128)
    potential = jnp.zeros((len(t), len(x)), dtype=jnp.float64)
    for iteration in range(0, len(t)):
        potential = potential.at[iteration].set(V(x, t[iteration]))

    psi = psi.at[0].set(waveFunctionGenerator(x, 0))

    for iteration in range(0, simConstants["tCount"]):
        threadStatus["percent"] = 100 * iteration / simConstants["tCount"]
        time = t[iteration]
        A = crankNicolson.computeLeft(
            x,
            psi[iteration],  # psi
            potential[iteration + 1],
            simConstants["dx"],
            simConstants["dt"],
            simConstants["mass"],
            simConstants["hbar"],
            simConstants["g"],
        )

        B = crankNicolson.computeRight(
            x,
            psi[iteration],
            potential[iteration],
            simConstants["dx"],
            simConstants["dt"],
            simConstants["mass"],
            simConstants["hbar"],
            simConstants["g"],
        )
        right = B @ psi[iteration]
        psi = psi.at[iteration + 1].set(jnp.linalg.solve(A, right))

    # Save the simulation
    jnp.save(os.path.join(constants.SIMULATIONS_FOLDER, "results", threadStatus["simulation_id"] + ".npy"), psi)
    threadStatus["finished"] = True
    threadStatus["percent"] = 100
    threadStatus["status"] = "finished"
