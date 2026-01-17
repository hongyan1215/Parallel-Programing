# NWChem Build and Run Instructions

This document describes the complete, step-by-step procedure to compile and run **NWChem** on the Taiwania3 cluster for HW1-Bonus.  
All steps have been tested with the provided environment and are fully reproducible.

---

## 1. Prepare Environment

First, log in to the Taiwania3 login node and allocate a build session:
Then load the Intel oneAPI compilers and TA-provided OpenMPI:

```bash
source /work/b11902043/PP25/intel/oneapi/setvars.sh --force
export OMPI_HOME=/work/b11902043/PP25/openmpi
export PATH="$OMPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OMPI_HOME/lib:$LD_LIBRARY_PATH"
```

## 2. Download NWChem

Clone the official NWChem repository into your working directory:

```bash
cd ~/hw1bonus
git clone https://github.com/nwchemgit/nwchem.git
export NWCHEM_TOP=$HOME/hw1bonus/nwchem
```

## 3. Set Build Environment Variables

Configure environment variables for the NWChem build:

```bash
export NWCHEM_TARGET=LINUX64
export NWCHEM_MODULES=qm
export USE_MPI=y
export USE_MPIF=y
export USE_MPIF4=y
export ARMCI_NETWORK=MPI-PR
export USE_SCALAPACK=y
export USE_OPENMP=y
export FC=mpifort
export CC=mpicc
export CXX=mpicxx
```

## 4. Build Global Arrays (GA) and ARMCI-MPI

Inside the tools directory:

```bash
cd $NWCHEM_TOP/src/tools
./get-tools-github
MPICC=mpicc ./install-armci-mpi
```

## 5. Configure and Build NWChem

Run configuration and compilation:

```bash
cd $NWCHEM_TOP/src
make nwchem_config
make -j$(nproc) FFLAGS+=" -i8 -diag-disable=10448" CFLAGS+=" -diag-disable=10441"
```

After compilation, the executable will be generated at `$NWCHEM_TOP/bin/LINUX64/nwchem`.

## 6. Prepare Run Script

Create a batch script `hw1-bonus.sh` for Slurm:

```bash
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

export NWCHEM_TOP=/home/<username>/hw1bonus/nwchem

source /work/b11902043/PP25/intel/oneapi/setvars.sh --force
export OMPI_HOME=/work/b11902043/PP25/openmpi
export PATH="$OMPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OMPI_HOME/lib:$LD_LIBRARY_PATH"

export NWCHEM_TARGET=LINUX64
```