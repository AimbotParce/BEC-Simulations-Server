"""
Main program for the simulation api
"""

import os
import sys

from flask import Flask, jsonify, request
from simulation import create_simulation

app = Flask(__name__)


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Simulate the given data
    """
    data = request.get_json()
    return jsonify({"message": "Simulation started", "ok": True})


@app.route("/api/create", methods=["POST"])
def create():
    """
    Create a new simulation
    """
    data = request.get_json()
    response = create_simulation(data)
    return response


if __name__ == "__main__":
    app.run(debug=True)
