#!/usr/bin/env python
from mpi4py import MPI
from model import model
from ansatz import ansatzSinglePool
import argparse,sys,warnings



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--hmode", type=int, default=0,
        help="hmode. 0: h in mo; 1: h in ao. dflt: 0.")
parser.add_argument("-c", "--rcut", type=float, default=1.e-2,
        help="McLachlan distance cut-off. dflt: 0.01")
parser.add_argument("-f", "--fcut", type=float, default=1.e-2,
        help="invidual unitary cut-off ratio. dflt: 0.01")
parser.add_argument("-m", "--maxadd", type=int, default=5,
        help="Max. allowed unitaries to be added at one iteration. dflt: 5.")
parser.add_argument("-n", "--maxntheta", type=int, default=-1,
        help="Max. total allowed unitaries to be added. dflt: -1 (unlimited).")
parser.add_argument("-g", "--gsdegeneracy", type=int, default=1,
        help="Ground state degeneracy. dflt: 1.")
parser.add_argument("--eigvals", type=int, default=7,
        help="Eigenvalues to be calculated by ED. dflt: 7.")
parser.add_argument('--noarpack', action='store_true',
        help='Use eigh instead of arpack. dflt: False.')
parser.add_argument("-b", "--bound", type=float, default=10,
        help="Bounds for dtheta/dt: [-b, b]. dflt: 10")
parser.add_argument("--delta", type=float, default=1e-4,
        help="Tikhonov parameter. dflt: 1e-4. Nagative value switch on lsq.")
parser.add_argument("-v", "--vtol", type=float, default=1e-4,
        help="Tolerance for grandient. dflt: 1e-4.")
parser.add_argument("-t", "--dt", type=float, default=0.02,
        help="Time step size. dflt: 0.02")
parser.add_argument("-N", "--num_qubits", type=int, default=8,
        help="Number of qubits to generate filename.")
parser.add_argument("--g_tf", type=float, default=0.1,
        help="Transverse field to generate filename.")
args = parser.parse_args()

filename = "N" + str(args.num_qubits) + "g" + str(args.g_tf)

mdl = model(hmode=args.hmode, filename = filename)


ans = ansatzSinglePool(mdl,
        rcut=args.rcut,
        fcut=args.fcut,
        max_add=args.maxadd,
        maxntheta=args.maxntheta,
        bounds=[-args.bound, args.bound],
        dt=args.dt,
        delta=args.delta,
        vtol=args.vtol,
        filename=filename,
        )

comm = MPI.COMM_WORLD
m_rank = comm.Get_rank()


if m_rank == 0:
    versions = sys.version.split()[0].split('.')
    if int(versions[0]) != 3 or int(versions[1])!=8:
        warnings.warn("warning: mpi4py might require python3.8")

w, v = mdl.get_loweste_states(eigvals=args.eigvals, sparse=not args.noarpack)
if m_rank == 0:
    print("lowest energies: ", w)
    print('reference state energy:', mdl.get_h_expval(ans._ref_state))
    print('initial state energy:', mdl.get_h_expval(ans.get_state()))
    vec = ans.get_state()
    fidelity = 0
    for i in range(args.gsdegeneracy):
        fidelity += abs(vec.dot(v[i].data.as_ndarray().conj()))**2
    print("initial fidelity =", fidelity, flush=True)

ans.run()

if m_rank == 0:
    print("lowest energies: ", w)
    print("costs:", ans._e)
    vec = ans.get_state()
    fidelity = 0
    for i in range(args.gsdegeneracy):
        fidelity += abs(vec.dot(v[i].data.as_ndarray().conj()))**2
    print("fidelity =", fidelity, flush=True)
    ans.psave_ansatz_simp()
    ans.psave_ansatz_inp()
    ans.save_ansatz()
