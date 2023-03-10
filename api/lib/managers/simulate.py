import json
import os
import threading
from importlib.machinery import SourceFileLoader
from types import ModuleType

import jax
import jax.numpy as jnp

from .. import constants
from ..BEC_simulations.lib import constants as BECconstants
from ..BEC_simulations.lib.managers.crankNicolson import default as crankNicolson
from ..BEC_simulations.run import loadWaveFunctionAndPotential


def simulate(data):
    if not "name" in data:
        return {"message": "No simulation name given", "ok": False}

    # Check if a thread with the same name already exists
    for thread in threading.enumerate():
        if thread.name == data["name"]:
            return {"message": "Simulation already running", "ok": False}

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

    # Run the simulation in a new thread
    thread = threading.Thread(
        name=data["name"], target=__runSimulation, args=(simulationConstants, waveFunctionGenerator, V)
    )

    return {"message": "Simulation started", "ok": True}


def __runSimulation(simConstants, waveFunctionGenerator, V):

    # Run the simulation
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
    jnp.save(os.path.join(constants.SIMULATIONS_FOLDER, "psi.npy"), psi)
