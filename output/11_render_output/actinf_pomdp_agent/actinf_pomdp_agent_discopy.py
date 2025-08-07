#!/usr/bin/env python3
"""
DisCoPy categorical diagram code for Classic Active Inference POMDP Agent v1
Generated from GNN specification
"""

from discopy import *
from discopy.quantum import *
import numpy as np

def create_classic active inference pomdp agent v1_diagram():
    """Create a DisCoPy categorical diagram for Classic Active Inference POMDP Agent v1."""
    
    # Define types
    State = Ty('State')
    Observation = Ty('Observation')
    Action = Ty('Action')
    
    # Create basic boxes
    transition = Box('transition', State @ Action, State)
    observation = Box('observation', State, Observation)
    
    # Create diagram
    diagram = (State @ Action >> transition >> State >> observation >> Observation)
    
    return diagram

def create_classic active inference pomdp agent v1_quantum_diagram():
    """Create a quantum-inspired diagram for Classic Active Inference POMDP Agent v1."""
    
    # Define quantum types
    Qubit = Ty('Qubit')
    
    # Create quantum gates
    hadamard = Box('H', Qubit, Qubit)
    cnot = Box('CNOT', Qubit @ Qubit, Qubit @ Qubit)
    
    # Create quantum circuit
    circuit = (Qubit @ Qubit >> cnot >> Qubit @ Qubit)
    
    return circuit

def run_classic active inference pomdp agent v1_simulation():
    """Run DisCoPy simulation for Classic Active Inference POMDP Agent v1."""
    
    # Create diagrams
    classical_diagram = create_classic active inference pomdp agent v1_diagram()
    quantum_diagram = create_classic active inference pomdp agent v1_quantum_diagram()
    
    print("Classical diagram:")
    print(classical_diagram)
    print("\nQuantum diagram:")
    print(quantum_diagram)
    
    return classical_diagram, quantum_diagram

if __name__ == "__main__":
    classical, quantum = run_classic active inference pomdp agent v1_simulation()
    print("\nDisCoPy simulation completed successfully")
