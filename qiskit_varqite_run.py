"""
This script utilizes module "qiskit_varqite_main_class.py" to
perform VarQITE and ITE simulations for the TFIM using Qiskit.

Running "python qiskit_varqite_run.py" would run the code on the head node with
default parameters (see below).
Otherwise, the parameters can be specified in the command line or in the job
generating script to run the code on computing nodes
(see "qiskit_varqite_cluster_submission.py").
"""

import argparse
import os
import pickle
import sys

import qiskit_varqite_main_class
from qiskit_varqite_main_class import Noisy_varqite_qiskit_tfim


def none_or_str(value):
    """
    Function used to assign the type None to a command line variable if the
    command line reads "None", str otherwise.
    """
    if value == 'None':
        return None
    return value


def none_or_float(value):
    """
    Function used to assign the type None to a command line variable if the
    command line reads "None", float otherwise.
    """
    if value == 'None':
        return None
    return float(value)


def none_or_bool(value):
    """
    Function used to assign the type None to a command line variable if the
    command line reads "None", bool otherwise.
    """
    if value == 'None':
        return None
    return bool(value)


"""
Reading out parameters from the command line.
"""
parser = argparse.ArgumentParser(
    description = "Perform VarQITE and ITE simulations for TFIM using Qiskit"
)
parser.add_argument(
    "-n",
    "--num_qubits",
    type=int,
    default=2,
    metavar='\b',
    help="number of qubits"
)
parser.add_argument(
    "-g",
    "--g",
    type=float,
    default=0.1,
    metavar='\b',
    help="transverse field"
)
parser.add_argument(
    "-t",
    "--time",
    type=float,
    default=5.0,
    metavar='\b',
    help="total simulation (imaginary) time"
)
parser.add_argument(
    "--estimator_type",
    type=str,
    choices=["noiseless", "noisy"],
    default="noiseless",
    metavar='\b',
    help="estimator type"
)
parser.add_argument(
    "--noise_type",
    type=none_or_str,
    choices=[None, "X", "depolarizing"],
    default=None,
    help="noise type"
)
parser.add_argument(
    "-p",
    "--p",
    type=none_or_float,
    default=None,
    metavar='\b',
    help="noise probability"
)
parser.add_argument(
    "--estimator_approximation",
    type=none_or_bool,
    choices=[None, True, False],
    default=None,
    metavar='\b',
    help="estimator approximation flag"
)
parser.add_argument(
    "--estimator_method",
    type=none_or_str,
    choices=[None, "density_matrix", "statevector"],
    default=None,
    metavar='\b',
    help="estimator method"
)
parser.add_argument(
    "--num_cpus",
    type=int,
    default=1,
    metavar='\b',
    help="number of CPUs"
)
args = parser.parse_args()


"""
Initializing an instance of the class.
"""
varqite_obj = Noisy_varqite_qiskit_tfim(args.num_qubits)

"""
Performing VarQITE.
"""
evolution_result_varqite = varqite_obj.qiskit_varqite(
    args.g,
    args.time,
    args.estimator_type,
    args.noise_type,
    args.p,
    args.estimator_approximation,
    args.estimator_method
)

"""
Performing ITE.
"""
time_step=0.01
evolution_result_ite = varqite_obj.qiskit_ite(
    args.g,
    args.time,
    time_step
)

evolution_result = dict()
evolution_result["varqite"] = evolution_result_varqite
evolution_result["ite"] = evolution_result_ite

file_dir = os.path.dirname(os.path.abspath(__file__))

"""
Saving evolution_result dictionary containing desired objects to a file.
"""
with open(file_dir +
        '/data/qiskit_varqite_L%s_g%s_time%s_%s_noise%s_p%s_approx%s_%s.pkl' % (
            args.num_qubits,
            args.g,
            args.time,
            args.estimator_type,
            args.noise_type,
            args.p,
            args.estimator_approximation,
            args.estimator_method
            ), 'wb') as outp:
    pickle.dump(evolution_result, outp, pickle.HIGHEST_PROTOCOL)
