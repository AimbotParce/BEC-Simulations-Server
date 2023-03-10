import os
import re
import sys

from flask import jsonify

from .. import constants


def create_simulation(data):
    name = data.get("simulation_name")
    if not name:
        return jsonify({"message": "Missing name", "ok": False})

    if os.path.exists(os.path.join(constants.SIMULATIONS_FOLDER, name)):
        return jsonify({"message": "Simulation already exists", "ok": False})

    if not "wave_function" in data:
        return jsonify({"message": "Missing wave function", "ok": False})

    if not "potential" in data:
        return jsonify({"message": "Missing potential", "ok": False})

    os.mkdir(os.path.join(constants.SIMULATIONS_FOLDER, name))
    # Copy the basic simulation from api/lib/simulation.py to the new simulation folder
    basicSimu = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulationPattern.py"))

    with open(basicSimu, "r") as f:
        simulationContent = f.read()

    # Using regex to replace anything between <...> in simulationContent with the corresponding data
    pattern = re.compile(r"<(.*?)>")
    for match in pattern.finditer(simulationContent):
        if match.group(1) in data:
            simulationContent = simulationContent.replace(match.group(0), data[match.group(1)])
        else:
            simulationContent = simulationContent.replace(match.group(0), "")

    with open(os.path.join(constants.SIMULATIONS_FOLDER, name, "simulation.py"), "w") as f:
        f.write(simulationContent)

    with open(os.path.join(constants.SIMULATIONS_FOLDER, name, "constants.py"), "w") as f:
        if "constants" in data:
            f.write(data["constants"])

    # Create a directory for the results
    os.mkdir(os.path.join(constants.SIMULATIONS_FOLDER, name, "results"))

    return jsonify({"message": "Simulation created", "ok": True})
