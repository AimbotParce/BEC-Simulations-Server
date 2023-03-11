import threading

from .simulate import threadStatus


def get_status(data):
    """
    Get the status of the simulation
    """
    if not "simulation_id" in data:
        return {"message": "No simulation id given", "ok": False}

    if not data["simulation_id"] in threadStatus:
        return {"message": "Simulation does not exist", "ok": False}

    percent = threadStatus[data["simulation_id"]]["percent"]
    finished = threadStatus[data["simulation_id"]]["finished"]
    status = threadStatus[data["simulation_id"]]["status"]

    if finished:
        return {
            "message": "Simulation finished",
            "ok": True,
            "percent": percent,
            "finished": finished,
            "status": status,
        }

    return {"message": "Simulation running", "ok": True, "percent": percent, "finished": finished, "status": status}
