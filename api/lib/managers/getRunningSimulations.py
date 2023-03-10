from flask import jsonify

from .simulate import threadStatus


def get_running_simulations(data):
    """
    Get all running simulations
    """

    return jsonify({"ok": True, "message": "List of running simulations", "simulations": threadStatus})
