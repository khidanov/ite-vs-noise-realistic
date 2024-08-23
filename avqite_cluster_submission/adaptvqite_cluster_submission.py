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


set_of_params=[(num_qubits,
                round(g,3))
                for num_qubits in [8,10,12]
                for g in [0.2,0.3,0.5,0.6,0.8,0.9,1.1,1.2,1.3,1.4]
                ]

file_dir = os.path.dirname(os.path.abspath(__file__))

"""
For each parameter value, creating an sbatch job to execute.
Execution logs are saved into a directory logs/.
"""
CPUs = 24

for params in set_of_params:
    (num_qubits,
    g) = params
    if g < 1.0:
        gs_deg = 2
    else:
        gs_deg = 1
    job_file = "adaptvqite_cluster_submission_job.sbatch"
    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -t 20:00:00\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --ntasks-per-node=24\n")
        fh.writelines("#SBATCH --cpus-per-task=1\n")
        fh.writelines("#SBATCH --mem-per-cpu=4G\n")
        fh.writelines("#SBATCH --partition=dense\n")
        fh.writelines("#SBATCH --hint=compute_bound\n")
        fh.writelines("#SBATCH -o " + file_dir + "/out_err/" + "N" + str(num_qubits) + "g" + str(g) + "adaptvqite_out\n")
        fh.writelines("#SBATCH -e " + file_dir + "/out_err/" + "N" + str(num_qubits) + "g" + str(g) + "adaptvqite_err\n")
        fh.writelines("#SBATCH --job-name=\"adaptvqiteN"+str(num_qubits)+"g"+str(g)+"\"\n")
        # fh.writelines("#SBATCH --mail-user=\n")
        # fh.writelines("#SBATCH --mail-type=FAIL\n")
        # fh.writelines("export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        # fh.writelines("export OPENBLAS_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        # fh.writelines("export MKL_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        # fh.writelines("module load python py-numpy py-scipy py-joblib"
        #     "ml-gpu/20230427\n")
        fh.writelines("{ time mpirun -n 24 python " +
                        file_dir + "/run.py -g %s -N %s --g_tf %s " % ((gs_deg,) + params)  +
                        (" ; } 2> logs/time_adaptvqite_N%sg%s_dense.txt \n") % params
                    )
    """
    Executing the job.
    """
    os.system("sbatch %s" %job_file)

"""
Deleting the .sbatch file in the end.
"""
os.remove(job_file)
