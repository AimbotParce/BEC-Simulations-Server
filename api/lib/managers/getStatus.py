import threading

from .simulate import threadStatus


def get_status(data):
    """
    Get the status of the simulation
    """
    if not "name" in data:
        return {"message": "No simulation name given", "ok": False, "percent": 0, "finished": False}

    if not data["name"] in threadStatus:
        return {"message": "Simulation not running", "ok": False, "percent": 0, "finished": False}

    percent = threadStatus[data["name"]]["percent"]
    finished = threadStatus[data["name"]]["finished"]

    return {"message": "Simulation running", "ok": True, "percent": percent, "finished": finished}
