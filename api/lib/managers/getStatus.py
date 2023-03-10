import threading


def get_status(data):
    """
    Get the status of the simulation
    """
    if not "name" in data:
        return {"message": "No simulation name given", "ok": False}

    # Check if a thread with the same name already exists
    runningThread = None
    for thread in threading.enumerate():
        if thread.name == data["name"]:
            runningThread = thread
            break

    if runningThread is None:
        return {"message": "Simulation not running", "ok": False}
