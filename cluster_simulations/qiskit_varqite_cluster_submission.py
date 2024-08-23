"""
This script generates job scripts to run "qiskit_varqite_run.py" on
multiple computing nodes on a cluster.
The parameters for "qiskit_varqite_run.py" are specified in the
"set_of_params" list below.
Separate job is created and executed for each parameter set.
"""

import numpy as np
import os

"""
Specifying parameter values for "qiskit_varqite_run.py"
"""
# set_of_params=[(num_qubits,
#                 round(g,3),
#                 time,
#                 estimator_type,
#                 noise_type,
#                 p,
#                 estimator_approximation,
#                 estimator_method)
#                 for num_qubits in [8,10,12]
#                 for g in np.linspace(0.1,2.0,20)
#                 for time in [5.0]
#                 for estimator_type in ["noisy"]
#                 for noise_type in ["X"]
#                 for p in [0.0]
#                 for estimator_approximation in [True]
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
                for num_qubits in [8,10,12]
                for g in np.linspace(0.1,2.0,20)
                for time in [5.0]
                for estimator_type in ["noiseless"]
                for noise_type in [None]
                for p in [None]
                for estimator_approximation in [None]
                for estimator_method in [None]
                ]


file_dir = os.path.dirname(os.path.abspath(__file__))

"""
For each parameter value, creating an sbatch job to execute.
Execution logs are saved into a directory logs/.
"""
CPUs = 24
for params in set_of_params:
    (num_qubits,
    g,
    BC,
    time,
    estimator_type,
    noise_type,
    p,
    estimator_approximation,
    estimator_method) = params
    job_file = "qiskit_varqite_cluster_submission_job.sbatch"
    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -t 20:00:00\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --ntasks-per-node=1\n")
        fh.writelines("#SBATCH --cpus-per-task="+str(CPUs)+"\n")
        fh.writelines("#SBATCH --mem-per-cpu=4G\n")
        fh.writelines("#SBATCH --partition=volta\n")
        fh.writelines("#SBATCH --hint=compute_bound\n")
        fh.writelines("#SBATCH -o " + file_dir + "/qiskit_varqite_out\n")
        fh.writelines("#SBATCH -e " + file_dir + "/qiskit_varqite_err\n")
        fh.writelines("#SBATCH --job-name=\"L"+str(num_qubits)+"g"+str(g)+
                        "p"+str(p)+"\"\n")
        # fh.writelines("#SBATCH --mail-user=\n")
        # fh.writelines("#SBATCH --mail-type=FAIL\n")
        fh.writelines("export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        fh.writelines("export OPENBLAS_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        fh.writelines("export MKL_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        # fh.writelines("module load python py-numpy py-scipy py-joblib"
        #     "ml-gpu/20230427\n")
        fh.writelines("{ time python " +
                        file_dir +
                        ("/qiskit_varqite_run.py "
                        "-n %s "
                        "-g %s "
                        "-bc %s "
                        "-t %s "
                        "--estimator_type %s "
                        "--noise_type %s "
                        "-p %s "
                        "--estimator_approximation %s "
                        "--estimator_method %s "
                        "--num_cpus %s ")
                        % (params + (CPUs,)) +
                        (" ; } 2> logs/time_qiskit_varqite_L%s_g%s_%s_time%s_%s_"
                        "noise%s_p%s_approx%s_%s.txt \n") % params
                    )

    """
    Executing the job.
    """
    os.system("sbatch %s" %job_file)

"""
Deleting the .sbatch file in the end.
"""
os.remove(job_file)
