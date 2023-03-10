"""
Main program for the simulation api
"""

import os
import sys

import lib.managers as managers
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Simulate the given data
    """
    data = request.get_json()
    response = managers.simulate(data)
    return response


@app.route("/api/create", methods=["POST"])
def create():
    """
    Create a new simulation
    """
    data = request.get_json()
    response = managers.create_simulation(data)
    return response


if __name__ == "__main__":
    app.run(debug=True)
