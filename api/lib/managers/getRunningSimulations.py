import threading

from flask import jsonify

from .simulate import threadStatus


def list_simulations(data):
    """
    Get all running simulations
    """

    return jsonify(
        {
            "ok": True,
            "message": "List of running simulations",
            "open_threads": len(threading.enumerate()),
            "simulations": threadStatus,
        }
    )
