# Tutorial for LSF

```bash
# submit job
bsub < job.sh
# check job status
bjobs
# check job output
bpeek <job_id>
# cancel job
bkill <job_id>
# check queue of nodes
nodestat -F hpc
nodestat -F -g gpua100
nodestat -F -g gpuv100
nodestat -F -g gpua40
nodestat -F -g gpua10
# check job start time
showstart <job_id>
```

```bash
cd ~ && du -h --max-depth=1 .
# https://www.hpc.dtu.dk/?page_id=59
# https://www.hpc.dtu.dk/?page_id=927#Scratch_usage
```

### template for batch jobs
ref: https://www.hpc.dtu.dk/?page_id=2759
### python package FAQ
ref: https://www.hpc.dtu.dk/?page_id=3678
### LSF job management
ref: https://www.hpc.dtu.dk/?page_id=1519

## HPC example

1. 4 x Lenovo ThinkSystem SD630 V2
    - 2x Intel Xeon Gold 6342 (24 core, 2.8 GHz)
    - 512GB memory (16 x 32GB DDR4-3200)
    - FDR-Infiniband
    - 10 GBit-Ethernet
    - 480 GB SSD
2. 28 x Lenovo ThinkSystem SD530
    - 2x Intel Xeon Gold 6226R (16 core, 2.90GHz)
    - 11 x 384GB memory (12 x 32GB DDR4-2933)
    - 17 x 768 GB memory (12 x 64GB DDR4-2933)
    - FDR-Infiniband
    - 10 GBit-Ethernet
    - 480 GB SSD

```bash
#!/bin/sh
#BSUB -J Whipping
#BSUB -q hpc
#BSUB -n 48
#BSUB -W 05:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "select[model == XeonGold6342]"
#BSUB -u baixianger@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o RL_%J.out
#BSUB -e RL_%J.err
```

```bash
#!/bin/sh
#BSUB -J Whipping
#BSUB -q hpc
#BSUB -n 24
#BSUB -W 05:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -u baixianger@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o RL_%J.out
#BSUB -e RL_%J.err
```
ref: https://www.hpc.dtu.dk/?page_id=2520 [nodelist]
ref: https://www.hpc.dtu.dk/?page_id=1416 [hpc batch job template]

## GPU example

> For being able to run code on the Nvidia A100 please make sure to compile your code with cuda 11.0 or newer.
- Nvidia:
    - 4 nodes with 2 x Tesla A100 PCIE 40 GB (owned by DTU Compute) – queuename: `gpua100`
    - 6 nodes with 2 x Tesla A100 PCIE 80 GB (owned by DTU Compute) – queuename: `gpua100`
    - 6 nodes with 2 x Tesla V100 16 GB (owned by DTU Compute&DTU Elektro) – queuename: `gpuv100`
    - 8 nodes with 2 x Tesla V100 32 GB (owned by DTU Compute&DTU Environment&DTU MEK) – queuename `gpuv100`
    - 1 nodes with 2 x Tesla A10 PCIE 24 GB (owned by DTU Compute) – queuename `gpua10`
    - 1 nodes with 2 x Tesla A40 48 GB with NVlink (owned by DTU Compute) – queuename `gpua40`
    - 3 nodes with 4 x Tesla V100 32 GB with NVlink (owned by DTU Compute) – queuename `gpuv100`
    - 2 nodes with 4 x TitanX (Pascal) – queuename: gputitanxpascal (retired)
    - 1 node with 4 x Tesla K80 – queuename: gpuk80 (retired)
    - 1 node with 2 x Tesla K40 – queuename: gpuk40 
- AMD
    - 1 node with 2 x AMD Radeon Instinct MI50 16 GB gpus – not on queue
    - 1 node with 2 x AMD Radeon Instinct MI25 16 GB gpus – queuename `gpuamd`

- Interactive node
    - 1 interactive V100-node reachable via `voltash`
    - 1 interactive V100-node with NVlink reachable via `sxm2sh`
    - 1 interactive A100-node with NVlink reachable via `a100sh`

```bash
#!/bin/sh
#BSUB -J Whipping
#BSUB -q gpuv100
#BSUB -n 32
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 05:00
#BSUB -R "rusage[mem=8GB]"
# XeonGold6242: 32 core
# XeonGold6142: 32 core
# XeonGold6126: 24 core 
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6242]"
# this is only for v100 with a 32Gb memory
#BSUB -R "select[gpu32gb]"
# this is only for v100 with NVlink
#BSUB -R "select[sxm2]"
#BSUB -u baixianger@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o RL_%J.out
#BSUB -e RL_%J.err
```
ref: https://www.hpc.dtu.dk/?page_id=2759