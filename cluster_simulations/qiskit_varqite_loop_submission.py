"""
Script that runs qiskit_varqite_wrapper.py \
on multiple nodes by creating multiple jobs, one job for each parameter value \
Each parameter value is passed to qiskit_varqite_wrapper.py, \
and separate qiskit_varqite_wrapper.py is run for \
each parameter value on its own node
"""

import os
import numpy as np

"setting up parameter values"

file_dir = os.path.dirname(os.path.abspath(__file__))

# set_of_params=[(num_qubits,
#                 round(g,3),
#                 time,
#                 estimator_type,
#                 noise_type,
#                 p,
#                 estimator_approximation,
#                 estimator_method)
#                 for num_qubits in [8]
#                 for g in [0.1]
#                 for time in [5.0]
#                 for estimator_type in ["noisy"]
#                 for noise_type in ["X"]
#                 for p in [0.0]
#                 for estimator_approximation in ['True']
#                 for estimator_method in ['density_matrix']
#                 ]


set_of_params=[(num_qubits,
                round(g,3),
                time,
                estimator_type,
                noise_type,
                p,
                estimator_approximation,
                estimator_method)
                for num_qubits in [8]
                for g in [0.1]
                for time in [5.0]
                for estimator_type in ["noiseless"]
                for noise_type in [None]
                for p in [None]
                for estimator_approximation in [None]
                for estimator_method in [None]
                ]

"creating sbatch job to run for each parameter value"
CPUs = 24
for params in set_of_params:
    (num_qubits,
    g,
    time,
    estimator_type,
    noise_type,
    p,
    estimator_approximation,
    estimator_method) = params
    job_file = "qiskit_varqite_loop_submission_job.sbatch"
    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -t 20:00:00\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --ntasks-per-node=1\n") #
        fh.writelines("#SBATCH --cpus-per-task="+str(CPUs)+"\n")
        fh.writelines("#SBATCH --mem-per-cpu=4G\n")
        fh.writelines("#SBATCH --partition=dense\n")
        fh.writelines("#SBATCH --hint=compute_bound\n")
        fh.writelines("#SBATCH -o " + file_dir + "/qiskit_varqite_out\n")
        fh.writelines("#SBATCH -e " + file_dir + "/qiskit_varqite_err\n")
        fh.writelines("#SBATCH --job-name=\"L"+str(num_qubits)+"g"+str(g)+"p"+str(p)+"\"\n")
        # fh.writelines("#SBATCH --mail-user=akhin@iastate.edu\n")
        # fh.writelines("#SBATCH --mail-type=FAIL\n")
        fh.writelines("export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        fh.writelines("export OPENBLAS_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        fh.writelines("export MKL_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        # fh.writelines("module load python py-numpy py-scipy py-joblib ml-gpu/20230427\n")
        fh.writelines("{ time python " + file_dir + "/qiskit_varqite_wrapper.py %s %s %s %s %s %s %s %s %s"
            % (params + (CPUs,)) +
            " ; } 2> logs/time_qiskit_varqite_L%s_g%s_time%s_%s_noise%s_p%s_approx%s_%s.txt \n" % params )

    "running the job"
    os.system("sbatch %s" %job_file)

os.remove(job_file)
