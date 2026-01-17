#!/bin/bash
#SBATCH --job-name=nwchem
#SBATCH --partition=ctest
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G            
#SBATCH --time=0:5:00
#SBATCH --account=ACD114118
#SBATCH --output=nwchem.log
#SBATCH --error=nwchem.err

# NWChem 路徑（改成你編好的）
export NWCHEM_TOP=/home/r14922156/hw1bonus/nwchem

# 載入 Intel oneAPI & MPI
source /work/b11902043/PP25/intel/oneapi/setvars.sh --force
export OMPI_HOME=/work/b11902043/PP25/openmpi
export PATH="$OMPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OMPI_HOME/lib:$LD_LIBRARY_PATH"

# NWChem build options
export NWCHEM_TARGET=LINUX64
export NWCHEM_MODULES=qm
export FC=mpifort
export OMP_NUM_THREADS=1

echo "Starting NWChem calculation..."

# 用 TA 提供的 input 絕對路徑（提交必須用這個）
mpirun -np $SLURM_NTASKS $NWCHEM_TOP/bin/LINUX64/nwchem /work/b11902043/PP25/hw1-bonus/inputs/w12_b3lyp_cc-pvtz_energy.nw
