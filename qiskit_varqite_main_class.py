"""
Module to perform VarQITE and ITE simulations for the TFIM using Qiskit

VarQITE can be simulated with and without noise
VarQITE algorithm uses ImaginaryMcLachlanPrinciple as a variational principle \
and qiskit_aer Estimator to simulate noisy gates

ITE is performed without noise using SciPyImaginaryEvolver

Packages information
____________________
qiskit version = 1.1.1
qiskit-aer version = 0.14.2
qiskit-algorithms version = 0.3.0
"""


import sys
import numpy as np
from typing import Union, Optional, List, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Estimator as QiskitEstimator
from qiskit.quantum_info import Statevector

from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit_algorithms import TimeEvolutionProblem
from qiskit_algorithms import VarQITE
from qiskit_algorithms import SciPyImaginaryEvolver
from qiskit_algorithms.gradients import ReverseEstimatorGradient, ReverseQGT

from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from qiskit_aer.primitives import Estimator as AerEstimator


class Noisy_varqite_qiskit_tfim:
    """
    Class for performing VarQITE and ITE simulations for the TFIM using Qiskit
    VarQITE simulation is performed using method qiskit_varqite
    ITE simulation is performed using method qiskit_ite

    Attributes
    __________
    num_qubits: int
        Number of qubits in the simulated system
    """
    def __init__(
        self,
        num_qubits: int
    ):
        self.num_qubits = num_qubits

    def set_tfim_H(
        self,
        g: float,
        J: float = 1.0
    ) -> "SparsePauliOp":
        """

        """
        ZZ_string_array = ["I"*i+"ZZ"+"I"*(self.num_qubits-2-i)  for i in range(self.num_qubits-1)]
        J_array = [-J for i in range(self.num_qubits-1)]

        X_string_array = ["I"*i+"X"+"I"*(self.num_qubits-1-i)  for i in range(self.num_qubits)]
        g_array = [g for i in range(self.num_qubits)]

        H = SparsePauliOp(ZZ_string_array + X_string_array, coeffs = J_array + g_array)

        return H

    def set_aux_ops(
        self,
        g: float,
        J: float = 1.0
    ) -> List["SparsePauliOp"]:
        ZZ_string_array = ["I"*i+"ZZ"+"I"*(self.num_qubits-2-i)  for i in range(self.num_qubits-1)]
        Z_string_array = ["I"*i+"Z"+"I"*(self.num_qubits-1-i)  for i in range(self.num_qubits)]
        mag_coef_array = [1/self.num_qubits for i in range(self.num_qubits)]

        ZZ = SparsePauliOp(ZZ_string_array, coeffs = [1/(self.num_qubits-1) for i in range(self.num_qubits-1)])
        mag = SparsePauliOp(Z_string_array, coeffs = mag_coef_array)
        mag2 =  mag.power(2).simplify()
        mag4 = mag.power(4).simplify()

        return [ZZ, mag2, mag4]

    def set_ansatz_and_init_param(
        self,
        init_param_values_const: float = np.pi / 2
    ) -> Tuple["EfficientSU2", dict["ParameterVectorElement",float]]:
        ansatz = EfficientSU2(self.num_qubits, reps=1)
        init_param_values = {}
        for i in range(len(ansatz.parameters)):
            init_param_values[ansatz.parameters[i]] = init_param_values_const
        return ansatz, init_param_values

    def set_noise_model(
        self,
        noise_type: Optional[str],
        p: Optional[float]
    ) -> "NoiseModel":

        if noise_type == None:
            noise_model = None
        else:
            noise_model = NoiseModel()
            if noise_type == "X":
                error_gate1 = pauli_error([('X',p), ('I', 1 - p)])
                error_gate2 = error_gate1.tensor(error_gate1)
            elif noise_type == "depolarizing":
                error_gate1 = depolarizing_error(p, 1)   #single-qubit depolarizing error
                error_gate2 = error_gate1.tensor(error_gate1)
            else:
                print("Error: not supported noise type")
                sys.exit()
            noise_model.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
            noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])
        return noise_model

    def set_estimator(
        self,
        estimator_type: str,
        noise_model,
        estimator_approximation: bool,
        estimator_method: str,
        shots: Optional[int],
    ) -> "Estimator":
        """
        estimator_method: "density_matrix", "statevector", ...

        """
        if estimator_type == "noiseless":
            estimator = QiskitEstimator()
        if estimator_type == "noisy":

            if estimator_approximation == True:
                estimator = AerEstimator(
                backend_options={
                    "method": estimator_method,
                    "noise_model": noise_model,
                },
                run_options = None,
                approximation = True
                )
            if estimator_approximation == False:
                estimator = AerEstimator(
                backend_options={
                    "method": estimator_method,
                    "noise_model": noise_model,
                },
                run_options={"shots": shots},
                )
        return estimator

    def qiskit_varqite(
        self,
        g: float,
        time: float,
        estimator_type: str = "noiseless",
        noise_type: Optional[str] = None,
        p: Optional[float] = None,
        estimator_approximation: Optional[bool] = None,
        estimator_method: Optional[str] = None,
        shots: Optional[int] = None,
        ReverseQGTFlag: bool = False,
        ReverseEstimatorGradientFlag: bool = False
    ) -> "VarQTEResult":
        if estimator_type == "noiseless" and (noise_type != None or
                                              p != None or
                                              shots != None or
                                              estimator_approximation != None or
                                              shots != None or
                                              estimator_method != None
                                             ):
            print("Error: for a noiseless estimator noise_type, p, estimator_approximation, "
                "estimator_method, and shots have to be None type")
            sys.exit()
        if estimator_type == "noisy" and (noise_type == None or
                                          p == None or
                                          estimator_approximation == None or
                                          estimator_method == None
                                         ):
            print("Error: for a noisy estimator noise_type, p, estimator_approximation, "
                "and estimator_method have to be non-None type")
            sys.exit()
        if estimator_approximation == False and shots == None:
            print("Error: if estimator_approximation == False, then number of shots has to be int")
            sys.exit()

        self.H = self.set_tfim_H(g)
        self.aux_ops = self.set_aux_ops(g) + [self.H]
        self.ansatz, self.init_param_values = self.set_ansatz_and_init_param()

        evolution_problem = TimeEvolutionProblem(
            self.H,
            time,
            aux_operators = self.aux_ops
        )
        var_principle = ImaginaryMcLachlanPrinciple(
            qgt = ReverseQGT() if ReverseQGTFlag else None,
            gradient = ReverseEstimatorGradient() if ReverseEstimatorGradientFlag else None
        )
        self.noise_model = self.set_noise_model(
            noise_type,
            p
        )
        self.estimator = self.set_estimator(
            estimator_type,
            self.noise_model,
            estimator_approximation,
            estimator_method,
            shots
        )
        var_qite = VarQITE(self.ansatz, self.init_param_values, var_principle, self.estimator)
        evolution_result = var_qite.evolve(evolution_problem)

        return evolution_result

    def qiskit_ite(
        self,
        g: float,
        time: float,
        time_step: float
    ) -> "TimeEvolutionResult":
        self.H = self.set_tfim_H(g)
        self.aux_ops = self.set_aux_ops(g) + [self.H]
        self.ansatz, self.init_param_values = self.set_ansatz_and_init_param()

        init_state = Statevector(self.ansatz.assign_parameters(self.init_param_values))
        evolution_problem = TimeEvolutionProblem(
            self.H,
            time,
            initial_state = init_state,
            aux_operators = self.aux_ops
        )
        evol = SciPyImaginaryEvolver(num_timesteps=int(time/time_step))
        sol = evol.evolve(evolution_problem)

        return sol
