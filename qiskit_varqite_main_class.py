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
import h5py
from typing import (
    List,
    Optional,
    Tuple,
    Union
)
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
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


    def set_ansatz(
        self,
        ansatz_type: str = "adaptvqite",
        EfficientSU2_reps: int = 1,
        **kwargs
    ) -> Union["QuantumCircuit", "EfficientSU2"]:
        """
        Generates a parameterized ansatz circuit.

        Parameters
        ----------
        ansatz_type : str
            Type of ansatz.
            Relevant options: "adaptvqite", "EfficientSU2".

        Returns
        -------
        ansatz : Union["QuantumCircuit", "EfficientSU2"]
            Parameterized ansatz circuit.

        """
        if ansatz_type == "adaptvqite":
            filename = "ansatz_example.h5"
            ansatz_adaptvqite, ref_state_adaptvqite = self.read_adaptvqite_ansatz(filename)

            if ref_state_adaptvqite == [1.+0.j]+[0.+0.j for i in range(len(ref_state_adaptvqite)-1)]:
                ansatz = QuantumCircuit(self._num_qubits)
            elif ref_state_adaptvqite == [0.+0.j for i in range(len(ref_state_adaptvqite)-1)]+[1.+0.j]:
                ansatz = QuantumCircuit(self._num_qubits)
                ansatz.x([i for i in range(self._num_qubits)])
            else:
                raise ImportError('Reference state is assumed to be either |00...0> or |11...1> here.')

            for i, pauli_string_bytes in enumerate(ansatz_adaptvqite):

                pauli_string = ansatz_adaptvqite[i].decode("utf-8")

                theta = Parameter("theta%s" % i)

                ansatz.append(self.pauli_rotation_gate(theta, pauli_string), range(self._num_qubits))

        elif ansatz_type == "EfficientSU2":
            ansatz = EfficientSU2(self._num_qubits, reps=1)
        else:
            raise ValueError("Not supported ansatz type type. Relevant options: 'adaptvqite', 'EfficientSU2'")
        return ansatz


    def read_adaptvqite_ansatz(
        self,
        filename: str
    ):
        """
        Reads the ansatz from a file resulting from adaptvqite calculation.

        Parameters
        ----------
        filename : str
            Name of a file containing the results of adaptvqite calculation.
            Has to be given in .h5 format.

        Returns
        -------
        ansatz_adaptvqite : List[bytes]
            List of byte strings representing Pauli strings entering the ansatz.
        ref_state_adaptvqite : List[complex128]
            Normalized reference state.
        """
        if filename[-3:] != '.h5':
            raise ImportError("Ansatz file should be given in .h5 format")

        with h5py.File(filename, "r") as f:
            ansatz_adaptvqite = list(f['ansatz_code'])
            # ngates_adaptvqite = f['ngates'][()]
            # params_adaptvqite = f['params'][()]
            ref_state_adaptvqite = list(f['ref_state'])

        return ansatz_adaptvqite, ref_state_adaptvqite


    def pauli_rotation_gate(
        self,
        theta,
        pauli_string: str,
    ):
        """
        Generates a Pauli string rotation gate.

        Parameters
        ----------
        theta : float or "qiskit.circuit.parameter.Parameter"
            Pauli rotation angle.

        Returns
        -------
        gate : Qiskit instruction
        """
        operator = SparsePauliOp(pauli_string)
        gate = PauliEvolutionGate(operator, time = theta)
        return gate


    def set_init_params(
        self,
        ansatz,
        init_param_values_const: float = np.pi / 2
    ) -> dict["ParameterVectorElement",float]:
        """
        Generates a dictionary of initial parameter values for the ansatz
        circuit.
        Here for simplicity all initial parameter values are assumed to be
        the same.

        Parameters
        ----------
        ansatz : "QuantumCircuit" or "EfficientSU2"
            Parameterized ansatz circuit.
        init_param_values_const : float
            Initial parameter values, assumed to be the same here.

        Returns
        -------
        init_param_values : dict["ParameterVectorElement", float]
            Dictionary of initial parameter values for the ansatz circuit.
        """
        init_param_values = {}
        for i in range(len(ansatz.parameters)):
            init_param_values[ansatz.parameters[i]] = init_param_values_const
        return init_param_values


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
                raise ValueError("Not supported noise type.")
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
            raise ValueError(
                "For a noiseless estimator noise_type, p, "
                "estimator_approximation, estimator_method, and shots have to "
                "be None type."
                )
        if estimator_type == "noisy" and (noise_type == None or
                                          p == None or
                                          estimator_approximation == None or
                                          estimator_method == None
                                         ):
            raise ValueError(
                "For a noisy estimator noise_type, p, "
                "estimator_approximation, and estimator_method have to be "
                "non-None type."
                )
        if estimator_approximation == False and shots == None:
            raise ValueError(
                "If estimator_approximation == False, then the number of shots "
                "has to be integer."
                )

        self.H = self.set_tfim_H(g)
        self.aux_ops = self.set_aux_ops() + [self.H]
        self.ansatz = self.set_ansatz(ansatz_type = "adaptvqite")
        self.init_param_values = self.set_init_params(self.ansatz)

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
        self.ansatz = self.set_ansatz()
        self.init_param_values = self.set_init_params(self.ansatz)

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
