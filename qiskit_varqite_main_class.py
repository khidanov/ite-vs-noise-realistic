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
import pickle
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
        BC: str,
        J: float = 1.0
    ) -> "SparsePauliOp":
        """
        Generates TFIM Hamiltonian in Qiskit "SparsePauliOp" format.

        Parameters
        ----------
        g : float
            Transverse field value (typically in the units of J).
        BC : periodic
            Boundary conditions.
            Relevant options: "periodic", "open".
        J : float
            Interaction strength value.
        """
        if BC == "open":
            ZZ_string_array = [
                "I"*i+"ZZ"+"I"*(self._num_qubits-2-i)
                for i in range(self._num_qubits-1)
            ]
            J_array = [-J for i in range(self._num_qubits-1)]
        elif BC == "periodic":
            ZZ_string_array = [
                "I"*i+"ZZ"+"I"*(self._num_qubits-2-i)
                for i in range(self._num_qubits-1)
            ] + ["Z"+"I"*(self._num_qubits-2)+"Z"]
            J_array = [-J for i in range(self._num_qubits)]
        else:
            raise ValueError(
                "Not supported boundary conditions. "
                "Possible options: 'periodic', 'open'."
            )
        X_string_array = [
            "I"*i+"X"+"I"*(self._num_qubits-1-i)
            for i in range(self._num_qubits)
        ]
        g_array = [g for i in range(self._num_qubits)]

        H = SparsePauliOp(
            ZZ_string_array + X_string_array, coeffs = J_array + g_array
        )

        return H


    def set_aux_ops(
        self,
        BC: str
    ) -> List["SparsePauliOp"]:
        """
        Generates a list of auxilary operators.
        Auxilary operators are generated to copmute their expectation values
        and, using the expectation values, to extract the Binder cumulant.
        The Hamiltonian itself is not added as an auxilary operator here.

        Parameters
        ----------
        BC : periodic
            Boundary conditions.
            Relevant options: "periodic", "open".

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
        if BC == "open":
            ZZ_string_array = [
                "I"*i+"ZZ"+"I"*(self._num_qubits-2-i)
                for i in range(self._num_qubits-1)
            ]
        elif BC == "periodic":
            ZZ_string_array = [
                "I"*i+"ZZ"+"I"*(self._num_qubits-2-i)
                for i in range(self._num_qubits-1)
            ] + ["Z"+"I"*(self._num_qubits-2)+"Z"]
        else:
            raise ValueError(
                "Not supported boundary conditions. "
                "Possible options: 'periodic', 'open'."
            )
        Z_string_array = [
            "I"*i+"Z"+"I"*(self._num_qubits-1-i)
            for i in range(self._num_qubits)
        ]
        mag_coef_array = [1/self._num_qubits for i in range(self._num_qubits)]

        ZZ = SparsePauliOp(
            ZZ_string_array,
            coeffs = [1/self._num_qubits for i in range(self._num_qubits)]
        )
        mag = SparsePauliOp(Z_string_array, coeffs = mag_coef_array)
        mag2 =  mag.power(2).simplify()
        mag4 = mag.power(4).simplify()

        return [ZZ, mag2, mag4]


    def set_ansatz(
        self,
        ansatz_type: str = "adaptvqite",
        EfficientSU2_reps: Optional[int] = None
    ):
        """
        Generates a parameterized ansatz circuit.

        Parameters
        ----------
        ansatz_type : str
            Type of ansatz.
            Relevant options: "adaptvqite", "EfficientSU2".
        EfficientSU2_reps : Optional[Int]
            Number of reps in the "EfficientSU2" ansatz.
            If ansatz_type="adaptvqite", then EfficientSU2_reps should be None.

        Returns
        -------
        ansatz : Union["QuantumCircuit", "EfficientSU2"]
            Parameterized ansatz circuit.
        init_param_values_dict : Dict[float64]
            Parameters (angles) of the ansatz.
            If ansatz_type="EfficientSU2", ansatz parameters are chosen to be
            a constant value of np.pi/2.
        """
        if ansatz_type == "adaptvqite":
            filename = "adaptvqite_ansatz/ansatz_inp"+self.filename+".pkle"
            (ansatz_adaptvqite,
             params_ansatz) = self.read_adaptvqite_ansatz(filename)
            #opening incar file to read out the reference state
            with open("incars/incar"+self.filename) as fp:
                incar_content = fp.read()
            if incar_content[-(3+self._num_qubits):-3] == "0"*self._num_qubits:
                ansatz = QuantumCircuit(self._num_qubits)
            # applying bit-flip if initial state is |11...1>
            elif incar_content[-(3+self._num_qubits):-3] == "1"*self._num_qubits:
                ansatz = QuantumCircuit(self._num_qubits)
                ansatz.x([i for i in range(self._num_qubits)])
            else:
                raise ImportError(
                    "Reference state is assumed to be either |00...0> or "
                    "|11...1> here."
                )

            "setting the ansatz"
            for i, pauli_string in enumerate(ansatz_adaptvqite):
                theta = Parameter("theta%s" % i)
                ansatz.append(
                    self.pauli_rotation_gate(theta, pauli_string),
                    range(self._num_qubits)
                )
            "setting the dictionary of initial parameter values"
            init_param_values_dict = {}
            for i in range(len(ansatz.parameters)):
                param_idx_sorted_alphabetically = int(
                    sorted([str(j) for j in range(len(ansatz.parameters))])[i]
                )    # to follow qiskit notation
                init_param_values_dict[ansatz.parameters[i]] = params_ansatz[
                    param_idx_sorted_alphabetically
                ]
        elif ansatz_type == "EfficientSU2":
            ansatz = EfficientSU2(self._num_qubits, reps = EfficientSU2_reps)
            init_param_values_dict = {}
            for i in range(len(ansatz.parameters)):
                init_param_values_dict[ansatz.parameters[i]] = np.pi/2
        else:
            raise ValueError(
                "Not supported ansatz type. Possible options: 'adaptvqite', "
                "'EfficientSU2'."
            )
        return ansatz, init_param_values_dict


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
            Has to be given in .pkle format.

        Returns
        -------
        ansatz_adaptvqite : List[str]
            List of Pauli strings entering the ansatz.
        params_adaptvqite : List[float64]
            Parameters (angles) of the ansatz.
        """
        if filename[-5:] != '.pkle':
            raise ImportError("Ansatz file should be given in .pkle format")

        with open(filename, 'rb') as inp:
            data_inp = pickle.load(inp)
            ansatz_adaptvqite = data_inp[0]
            params_adaptvqite = data_inp[1]

        return ansatz_adaptvqite, params_adaptvqite


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
        gate = PauliEvolutionGate(operator, time = theta/2)
        return gate

    def set_noise_model(
        self,
        noise_type: Optional[str],
        p: Optional[float]
    ) -> "NoiseModel":
        """
        Generates a noise model for a Qiksit estimator.

        Parameters
        ----------
        noise_type : Optional[str]
            Noise type.
            Relevant options: "X", "depolarizing" or None.
        p : Optional[float]
            Noise strength (error probability) 1>p>=0.

        Returns
        -------
        noise_model : "qiskit_aer.noise.noise_model.NoiseModel"
        """
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
        Generates an (in general) noisy Qiskit estimator for VarQITE.
        The estimator uses Qiskit Aer Simulator under the hood.

        Parameters
        ----------
        estimator_type : str
            Relevant options: "noiseless" of "noisy".
        noise_model : "qiskit_aer.noise.noise_model.NoiseModel"
            Qiskit Aer noise model
        estimator_approximation : bool
            Flag whether to use an approximation for the estimator.
            If True, it calculates expectation values with normal distribution
            approximation.
            If False, then the actiual measuement shots are taken and the shot
            noise is included (note that the number of shots has to be
            specified in this case).
        estimator_method : str
            Method for Qiskit Aer Simulator used in the Estimator.
            Relevant options: "density_matrix", "statevector".
        shots : Optional[int]
            Number of measurement shots to be used in the estimator.
            Needs to be specified only if estimator_approximation = False.

        Returns
        -------
        estimator : "qiskit.primitives.estimator.Estimator"
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
        BC: str,
        time: float,
        ansatz_type: str = "adaptvqite",
        EfficientSU2_reps: Optional[int] = None,
        estimator_type: str = "noiseless",
        noise_type: Optional[str] = None,
        p: Optional[float] = None,
        estimator_approximation: Optional[bool] = None,
        estimator_method: Optional[str] = None,
        shots: Optional[int] = None,
        RevQGTFlag: bool = False,
        RevEstGradFlag: bool = False
    ) -> "VarQTEResult":
        """
        Performs an (in general) noisy VarQITE simulation using Qiskit.

        Parameters
        ----------
        g : float
            Transverse field value.
        BC : str
            Boundary conditions for the TFIM.
        time : float
            Total VarQITE simulation time.
        ansatz_type : str
            Relevant options: "adaptvqite" or "EfficientSU2".
        EfficientSU2_reps : Optional[Int]
            Number of reps in the "EfficientSU2" ansatz.
            If ansatz_type="adaptvqite", then EfficientSU2_reps should be None.
        estimator_type : str
            Relevant options: "noiseless" of "noisy".
        noise_model : "qiskit_aer.noise.noise_model.NoiseModel"
            Qiskit Aer noise model
        p : Optional[float]
            Noise strength (error probability) 1>p>=0.
        estimator_approximation : bool
            Flag whether to use an approximation for the estimator.
            If True, it calculates expectation values with normal distribution
            approximation.
            If False, then the actiual measuement shots are taken and the shot
            noise is included (note that the number of shots has to be
            specified in this case).
        estimator_method : str
            Method for Qiskit Aer Simulator used in the Estimator.
            Relevant options: "density_matrix", "statevector".
        shots : Optional[int]
            Number of measurement shots to be used in the estimator.
            Needs to be specified only if estimator_approximation = False.
        RevQGTFlag : bool
            Flag whether to use classical ReverseQGT implementation of QGT.
            ReverseQGT implementation is based on statevector manipulations and
            scales exponentially with the number of qubits.
            However, for small system sizes it can be very fast compared to
            circuit-based gradients.
            For more, see "https://docs.quantum.ibm.com/api/qiskit/0.45/qiskit.\
                            algorithms.gradients.ReverseQGT"
        RevEstGradFlag : bool
            Flag whether to use classical ReverseEstimatorGradient
            calculation of the expectation gradients.
            ReverseEstimatorGradient implementation is based on statevector
            manipulations and scales exponentially with the number of qubits.
            However, for small system sizes it can be very fast compared to
            circuit-based gradients.
            For more, see "https://qiskit-community.github.io/qiskit-algorithms\
            /stubs/qiskit_algorithms.gradients.ReverseEstimatorGradient.html"

        Returns
        -------
        evolution_result : "qiskit_algorithms.time_evolvers.variational.\
                            var_qte_result.VarQTEResult"
            Evolution result object.
        """
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
        if ansatz_type == "adaptvqite" and EfficientSU2_reps != None:
            raise ValueError(
                "For adaptvqite ansatz the number of reps should be None."
            )
        if ansatz_type == "EfficientSU2" and EfficientSU2_reps == None:
            raise ValueError(
                "For EfficientSU2 ansatz the number of reps should be "
                "specified."
            )

        self.H = self.set_tfim_H(g, BC)
        self.aux_ops = self.set_aux_ops(BC) + [self.H]
        self.filename = "N%sg%s" % (self._num_qubits, g)
        self.ansatz, self.init_param_values = self.set_ansatz(
            ansatz_type,
            EfficientSU2_reps
        )
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
        BC: str,
        time: float,
        ansatz_type: str = "adaptvqite",
        EfficientSU2_reps: Optional[int] = None,
        time_step: float = 0.01
    ) -> "TimeEvolutionResult":
        """
        Performs a noiseless Trotterized ITE simulation using Qiskit
        SciPyImaginaryEvolver.

        Parameters
        ----------
        g : float
            Transverse field value.
        BC : str
            Boundary conditions for the TFIM.
        time : float
            Total VarQITE simulation time.
        ansatz_type : str
            Relevant options: "adaptvqite" or "EfficientSU2".
        EfficientSU2_reps : Optional[Int]
            Number of reps in the "EfficientSU2" ansatz.
        time_step : float
            Time step of the Trotterized ITE evolution.

        Returns
        -------
        sol : "qiskit_algorithms.time_evolvers.time_evolution_result.\
                TimeEvolutionResult"
            Evolution result object.
        """
        self.H = self.set_tfim_H(g, BC)
        self.aux_ops = self.set_aux_ops(BC) + [self.H]
        self.filename = "N%sg%s" % (self._num_qubits, g)
        self.ansatz, self.init_param_values = self.set_ansatz(
            ansatz_type,
            EfficientSU2_reps
        )
        self.init_state = Statevector(
            self.ansatz.assign_parameters(self.init_param_values)
        )
        evolution_problem = TimeEvolutionProblem(
            self.H,
            time,
            initial_state = self.init_state,
            aux_operators = self.aux_ops
        )
        evol = SciPyImaginaryEvolver(num_timesteps=int(time/time_step))
        sol = evol.evolve(evolution_problem)

        return sol
