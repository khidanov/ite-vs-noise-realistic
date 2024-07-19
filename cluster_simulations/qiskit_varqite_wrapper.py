import pickle
import sys
import qiskit_varqite_main_class
from qiskit_varqite_main_class import Noisy_varqite_qiskit_tfim


"reading out parameters from the command line"
num_qubits = int(sys.argv[1])
g = float(sys.argv[2])
time = float(sys.argv[3])
if sys.argv[4] == "None":
    estimator_type = None
else:
    estimator_type = sys.argv[4]
if sys.argv[5] == "None":
    noise_type = None
else:
    noise_type = sys.argv[5]
if sys.argv[6] == "None":
    p = None
else:
    p = float(sys.argv[6])
if sys.argv[7] == "None":
    estimator_approximation = None
else:
    estimator_approximation = bool(sys.argv[7])
if sys.argv[8] == "None":
    estimator_method = None
else:
    estimator_method = sys.argv[8]

'initializing an instance of the class'
varqite_obj = Noisy_varqite_qiskit_tfim(num_qubits)

evolution_result_varqite = varqite_obj.qiskit_varqite(
    g,
    time,
    estimator_type,
    noise_type,
    p,
    estimator_approximation,
    estimator_method
)

time_step=0.01
evolution_result_ite = varqite_obj.qiskit_ite(
    g,
    time,
    time_step
)

evolution_result = dict()
evolution_result["varqite"] = evolution_result_varqite
evolution_result["ite"] = evolution_result_ite

with open('/home/akhin/ite_noise_realistic/qiskit_varqite/data/qiskit_varqite_L%s_g%s_time%s_%s_noise%s_p%s_approx%s_%s.pkl' %
            (num_qubits,
            g,
            time,
            estimator_type,
            noise_type,
            p,
            estimator_approximation,
            estimator_method
            ), 'wb') as outp:
    pickle.dump(evolution_result, outp, pickle.HIGHEST_PROTOCOL)
