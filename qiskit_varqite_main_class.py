"""
This module is used to perform VarQITE and ITE simulations for the TFIM using
Qiskit.

VarQITE can be simulated with or without noise.
VarQITE algorithm uses ImaginaryMcLachlanPrinciple as a variational principle
and qiskit_aer Estimator to simulate noisy gates.

ITE is performed without noise using SciPyImaginaryEvolver.

Useful Qiskit examples:
-----------------------
1) VarQite (noiseless) and ITE:
    https://qiskit-community.github.io/qiskit-algorithms/tutorials/ \
    11_VarQTE.html
2) VQE w/ and w/o noise (example of setting up a noisy Qiskit Aer Estimator):
    https://qiskit-community.github.io/qiskit-algorithms/tutorials/ \
    03_vqe_simulation_with_noise.html
3) Building noise models with Qiskit Aer:
    https://qiskit.github.io/qiskit-aer/tutorials/3_building_noise_models.html
4) Qiskit Aer Simulator method options:
    https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html

Packages information:
---------------------
python >= 3.7
qiskit version = 1.1.1
qiskit-aer version = 0.14.2
qiskit-algorithms version = 0.3.0
"""

import numpy as np
from typing import (
    List,
    Optional,
    Tuple,
    Union
)
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Estimator as QiskitEstimator
from qiskit_algorithms.time_evolvers.variational import (
    ImaginaryMcLachlanPrinciple
)
from qiskit_algorithms import (
    SciPyImaginaryEvolver,
    TimeEvolutionProblem,
    VarQITE
)
from qiskit_algorithms.gradients import ReverseEstimatorGradient, ReverseQGT
from qiskit_aer.noise import (
    depolarizing_error,
    NoiseModel,
    pauli_error,
    QuantumError,
    ReadoutError,
    thermal_relaxation_error
)
from qiskit_aer.primitives import Estimator as AerEstimator


class Noisy_varqite_qiskit_tfim:
    """
    Class for performing VarQITE and ITE simulations for the TFIM using Qiskit.
    VarQITE simulation is performed using method qiskit_varqite.
    ITE simulation is performed using method qiskit_ite.

    Attributes
    ----------
    num_qubits : int
        Number of qubits in the simulated system.
    H : "SparsePauliOp"
        TFIM Hamiltonain of the system in the Qiskit "SparsePauliOp" format.
    aux_ops : List["SparsePauliOp"]
        List of auxilary operators in the Qiskit "SparsePauliOp" format.
        The expectation values of the auxilary operators are computed
        throughout the VarQITE/ITE simulations.
        H is included as one of the auxilary operators.
    ansatz : "EfficientSU2"
        Parameterized ansatz circuit.
        Here ansatz circuit is given in the Qiskit "EfficientSU2" format.
    init_param_values : dict["ParameterVectorElement", float]
        Initial parameter values for the parameterized ansatz circuit.
        Each key in the dictionary corresponds to an indiviudal parameter
        (angle) and is the Qiskit "ParameterVectorElement" format.
    """
    def __init__(
        self,
        num_qubits: int
    ):
        self._num_qubits = num_qubits


    def set_tfim_H(
        self,
        g: float,
        J: float = 1.0
    ) -> "SparsePauliOp":
        """
        Generates TFIM Hamiltonian in Qiskit "SparsePauliOp" format.

        Parameters
        ----------
        g : float
            Transverse field value (typically in the units of J).
        J : float
            Interaction strength value.
        """
        ZZ_string_array = [
            "I"*i+"ZZ"+"I"*(self._num_qubits-2-i)
            for i in range(self._num_qubits-1)
        ]
        J_array = [-J for i in range(self._num_qubits-1)]

        X_string_array = [
            "I"*i+"X"+"I"*(self._num_qubits-1-i)
            for i in range(self._num_qubits)
        ]
        g_array = [g for i in range(self._num_qubits)]

        H = SparsePauliOp(
            ZZ_string_array + X_string_array,coeffs = J_array + g_array
        )

        return H


    def set_aux_ops(
        self
    ) -> List["SparsePauliOp"]:
        """
        Generates a list of auxilary operators.
        Auxilary operators are generated to copmute their expectation values
        and, using the expectation values, to extract the Binder cumulant.
        The Hamiltonian itself is not added as an auxilary operator here.

        Returns
        -------
        List of auxilary operators, each in Qiskit "SparsePauliOp" format.
            ZZ : "SparsePauliOp"
                \sum_{i}Z_iZ_{i+1}/(num_qubits-1)
            mag2 : "SparsePauliOp"
                total magnetization density squared
            mag4 : "SparsePauliOp"
                total magnetization density to the 4th power
        """
        ZZ_string_array = [
            "I"*i+"ZZ"+"I"*(self._num_qubits-2-i)
            for i in range(self._num_qubits-1)
        ]
        Z_string_array = [
            "I"*i+"Z"+"I"*(self._num_qubits-1-i)
            for i in range(self._num_qubits)
        ]
        mag_coef_array = [1/self._num_qubits for i in range(self._num_qubits)]

        ZZ = SparsePauliOp(
            ZZ_string_array,
            coeffs = [1/(self._num_qubits-1) for i in range(self._num_qubits-1)]
        )
        mag = SparsePauliOp(Z_string_array, coeffs = mag_coef_array)
        mag2 =  mag.power(2).simplify()
        mag4 = mag.power(4).simplify()

        return [ZZ, mag2, mag4]


    def set_ansatz_and_init_param(
        self,
        init_param_values_const: float = np.pi / 2
    ) -> Tuple["EfficientSU2", dict["ParameterVectorElement",float]]:
        """
        Generates a parameterized ansatz circuit and a dictionary of initial
        parameter values for the ansatz circuit.
        Here for simplicity all initial parameter values are assumed to be
        the same.

        Parameters
        ----------
        init_param_values_const : float
            Initial parameter values, assumed to be the same here.

        Returns
        -------
        ansatz : "EfficientSU2"
            Parameterized ansatz circuit.
        init_param_values : dict["ParameterVectorElement", float]
            Dictionary of initial parameter values for the ansatz circuit.
        """
        ansatz = EfficientSU2(self._num_qubits, reps=3)
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
                error_gate1 = depolarizing_error(p, 1)
                error_gate2 = error_gate1.tensor(error_gate1)
            else:
                raise ValueError("Not supported noise type")
            noise_model.add_all_qubit_quantum_error(
                error_gate1, ["u1", "u2", "u3"]
            )
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
        Returns estimator

        Parameters
        ----------
        estimator_method: str
            Method for Qiskit Aer Simulator used in the Estimator
            Relevant options: "density_matrix", "statevector"
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
        RevQGTFlag: bool = False,
        RevEstGradFlag: bool = False
    ) -> "VarQTEResult":
        if estimator_type == "noiseless" and (noise_type != None or
                                              p != None or
                                              shots != None or
                                              estimator_approximation != None or
                                              shots != None or
                                              estimator_method != None
                                             ):
            raise ValueError("For a noiseless estimator noise_type, p, "
                "estimator_approximation, estimator_method, and shots have to "
                "be None type")
        if estimator_type == "noisy" and (noise_type == None or
                                          p == None or
                                          estimator_approximation == None or
                                          estimator_method == None
                                         ):
            raise ValueError("For a noisy estimator noise_type, p, "
            "estimator_approximation, and estimator_method have to be non-None "
            "type")
        if estimator_approximation == False and shots == None:
            raise ValueError("If estimator_approximation == False, then the "
            "number of shots has to be int")

        self.H = self.set_tfim_H(g)
        self.aux_ops = self.set_aux_ops() + [self.H]
        self.ansatz, self.init_param_values = self.set_ansatz_and_init_param()

        evolution_problem = TimeEvolutionProblem(
            self.H,
            time,
            aux_operators = self.aux_ops
        )
        var_principle = ImaginaryMcLachlanPrinciple(
            qgt = ReverseQGT() if RevQGTFlag else None,
            gradient = ReverseEstimatorGradient() if RevEstGradFlag else None
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
        var_qite = VarQITE(
            self.ansatz,
            self.init_param_values,
            var_principle,
            self.estimator
        )
        evolution_result = var_qite.evolve(evolution_problem)

        return evolution_result


    def qiskit_ite(
        self,
        g: float,
        time: float,
        time_step: float
    ) -> "TimeEvolutionResult":
        self.H = self.set_tfim_H(g)
        self.aux_ops = self.set_aux_ops() + [self.H]
        self.ansatz, self.init_param_values = self.set_ansatz_and_init_param()

        init_state = Statevector(
            self.ansatz.assign_parameters(self.init_param_values)
        )
        evolution_problem = TimeEvolutionProblem(
            self.H,
            time,
            initial_state = init_state,
            aux_operators = self.aux_ops
        )
        evol = SciPyImaginaryEvolver(num_timesteps=int(time/time_step))
        sol = evol.evolve(evolution_problem)

        return sol
