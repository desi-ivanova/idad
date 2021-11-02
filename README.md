# Implicit Deep Adaptive Design (iDAD)
This code supports the NeurIPS paper 'Implicit Deep Adaptive Design: Policy-Based Experimental Design without Likelihoods'.

@article{ivanova2021implicit,
  title={Implicit Deep Adaptive Design: Policy-Based Experimental Design without Likelihoods},
  author={Ivanova, Desi R. and Foster, Adam and Kleinegesse, Steven and Gutmann, Michael and Rainforth, Tom},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}

## Computing infrastructure requirements
We have tested this codebase on Linux (Ubuntu x86_64) and MacOS (Big Sur v11.2.3) with Python 3.8.
To train iDAD networks, we recommend the use of a GPU. We used one GeForce RTX 3090 GPU on a machine with 126 GiB of CPU memory and 40 CPU cores.

## Installation
1. Ensure that Python and `conda` are installed.
1. Create and activate a new `conda` virtual environment as follows
```bash
conda create -n idad_code
conda activate idad_code
```
1. Install the correct version of PyTorch, following the instructions at [pytorch.org](https://pytorch.org/).
   For our experiments we used `torch==1.8.0` with CUDA version 11.1.
1. Install the remaining package requirements using `pip install -r requirements.txt`.
1. Install the torchsde package from its repository: `pip install git+https://github.com/google-research/torchsde.git`.

## MLFlow
We use `mlflow` to log metric and store network parameters. Each experiment run is stored in
a directory `mlruns` which will be created automatically. Each experiment is assigned a
numerical `<ID>` and each run gets a unique `<HASH>`. The iDAD networks will be saved in
`./mlruns/<ID>/<HASH>/artifacts`, which will be printed at the end of each training run.

## Location Finding Experiment

To train an iDAD network with the InfoNCE bound to locate 2 sources in 2D, using the approach in the paper, execute the command
```bash
python3 location_finding.py \
    --num-steps 100000 \
    --num-experiments=10 \
    --physical-dim 2 \
    --num-sources 2 \
    --lr 0.0005 \
    --num-experiments 10 \
    --encoding-dim 64 \
    --hidden-dim 512 \
    --mi-estimator InfoNCE \
    --device <DEVICE>
```

To train an iDAD network with the NWJ bound, using the approach in the paper, execute the command
```bash
python3 location_finding.py \
    --num-steps 100000 \
    --num-experiments=10 \
    --physical-dim 2 \
    --num-sources 2 \
    --lr 0.0005 \
    --num-experiments 10 \
    --encoding-dim 64 \
    --hidden-dim 512 \
    --mi-estimator NWJ \
    --device <DEVICE>
```

To run the static MINEBED baseline, use the following
```bash
python3 location_finding.py \
    --num-steps 100000 \
    --physical-dim 2 \
    --num-sources 2 \
    --lr 0.0001 \
    --num-experiments 10 \
    --encoding-dim 8 \
    --hidden-dim 512 \
    --design-arch static \
    --critic-arch cat \
    --mi-estimator NWJ \
    --device <DEVICE>
```

To run the static SG-BOED baseline, use the following
```bash
python3 location_finding.py \
    --num-steps 100000 \
    --physical-dim 2 \
    --num-sources 2 \
    --lr 0.0005 \
    --num-experiments 10 \
    --encoding-dim 8 \
    --hidden-dim 512 \
    --design-arch static \
    --critic-arch cat \
    --mi-estimator InfoNCE \
    --device <DEVICE>
```

To run the adaptive (explicit likelihood) DAD baseline, use the following
```bash
python3 location_finding.py \
    --num-steps 100000 \
    --physical-dim 2 \
    --num-sources 2 \
    --lr 0.0005 \
    --num-experiments 10 \
    --encoding-dim 32 \
    --hidden-dim 512 \
    --mi-estimator sPCE \
    --design-arch sum \
    --device <DEVICE>
```

To evaluate the resulting networks eun the following command
```bash
python3 eval_sPCE.py --experiment-id <ID>
```

To evaluate a random design baseline (requires no pre-training):
```bash
python3 baselines_locfin_nontrainable.py \
    --policy random \
    --physical-dim 2 \
    --num-experiments-to-perform 5 10 \
    --device <DEVICE>
```

To run the variational baseline (note: it takes a very long time), run:
```bash
python3 baselines_locfin_variational.py \
    --num-histories 128 \
    --num-experiments 10 \
    --physical-dim 2 \
    --lr 0.001 \
    --num-steps 5000\
    --device <DEVICE>
```
Copy `path_to_artifact` and pass it to the evaluation script:
```bash
python3 eval_sPCE_from_source.py \
    --path-to-artifact <path_to_artifact> \
    --num-experiments-to-perform 5 10 \
    --device <DEVICE>
```

## Pharmacokinetic Experiment

To train an iDAD network with the InfoNCE bound, using the approach in the paper, execute the command
```bash
python3 pharmacokinetic.py \
    --num-steps 100000 \
    --lr 0.0001 \
    --num-experiments 5 \
    --encoding-dim 32 \
    --hidden-dim 512 \
    --mi-estimator InfoNCE \
    --device <DEVICE>
```

To train an iDAD network with the NWJ bound, using the approach in the paper, execute the command
```bash
python3 pharmacokinetic.py \
    --num-steps 100000 \
    --lr 0.0001 \
    --num-experiments 5 \
    --encoding-dim 32 \
    --hidden-dim 512 \
    --mi-estimator NWJ \
    --gamma 0.5 \
    --device <DEVICE>
```

To run the static MINEBED baseline, use the following
```bash
python3 pharmacokinetic.py \
    --num-steps 100000 \
    --lr 0.001 \
    --num-experiments 5 \
    --encoding-dim 8 \
    --hidden-dim 512 \
    --design-arch static \
    --critic-arch cat \
    --mi-estimator NWJ \
    --device <DEVICE>
```

To run the static SG-BOED baseline, use the following
```bash
python3 pharmacokinetic.py \
    --num-steps 100000 \
    --lr 0.0005 \
    --num-experiments 5 \
    --encoding-dim 8 \
    --hidden-dim 512 \
    --design-arch static \
    --critic-arch cat \
    --mi-estimator InfoNCE \
    --device <DEVICE>
```

To run the adaptive (explicit likelihood) DAD baseline, use the following
```bash
python3 pharmacokinetic.py \
    --num-steps 100000 \
    --lr 0.0001 \
    --num-experiments 5 \
    --encoding-dim 32 \
    --hidden-dim 512 \
    --mi-estimator sPCE \
    --design-arch sum \
    --device <DEVICE>
```

To evaluate the resulting networks run the following command
```bash
python3 eval_sPCE.py --experiment-id <ID>
```

To evaluate a random design baseline (requires no pre-training):
```bash
python3 baselines_pharmaco_nontrainable.py \
    --policy random \
    --num-experiments-to-perform 5 10 \
    --device <DEVICE>
```

To evaluate an equal interval baseline (requires no pre-training):
```bash
python3 baselines_pharmaco_nontrainable.py \
    --policy equal_interval \
    --num-experiments-to-perform 5 10 \
    --device <DEVICE>
```

To run the variational baseline (note: it takes a very long time), run:
```bash
python3 baselines_pharmaco_variational.py \
    --num-histories 128 \
    --num-experiments 10 \
    --lr 0.001 \
    --num-steps 5000 \
    --device <DEVICE>
```
Copy `path_to_artifact` and pass it to the evaluation script:
```bash
python3 eval_sPCE_from_source.py \
    --path-to-artifact <path_to_artifact> \
    --num-experiments-to-perform 5 10 \
    --device <DEVICE>
```


## SIR experiment

For the SIR experiments, please first generate an initial training set and a test set:
```bash
python3 epidemic_simulate_data.py \
    --num-samples=100000 \
    --device <DEVICE>
```


To train an iDAD network with the InfoNCE bound, using the approach in the paper, execute the command
```bash
python3 epidemic.py \
    --num-steps 100000 \
    --num-experiments 5 \
    --lr 0.0005 \
    --hidden-dim 512 \
    --encoding-dim 32 \
    --mi-estimator InfoNCE \
    --design-transform ts \
    --device <DEVICE>
```
To train an iDAD network with the NWJ bound, execute the command
```bash
python3 epidemic.py \
    --num-steps 100000 \
    --num-experiments 5 \
    --lr 0.0005 \
    --hidden-dim 512 \
    --encoding-dim 32 \
    --mi-estimator NWJ \
    --design-transform ts \
    --device <DEVICE>
```

To run the static SG-BOED baseline, run
```bash
python3 epidemic.py \
    --num-steps 100000 \
    --num-experiments 5 \
    --lr 0.005 \
    --hidden-dim 512 \
    --encoding-dim 32 \
    --design-arch static \
    --critic-arch cat \
    --design-transform iid \
    --mi-estimator InfoNCE \
    --device <DEVICE>
```

To run the static MINEBED baseline, run
```bash
python3 epidemic.py \
    --num-steps 100000 \
    --num-experiments 5 \
    --lr 0.001 \
    --hidden-dim 512 \
    --encoding-dim 32 \
    --design-arch static \
    --critic-arch cat \
    --design-transform iid \
    --mi-estimator NWJ \
    --device <DEVICE>
```

To train a critic with random designs (to evaluate the random design baseline):
```bash
python3 epidemic.py \
    --num-steps 100000 \
    --num-experiments 5 \
    --lr 0.005 \
    --hidden-dim 512 \
    --encoding-dim 32 \
    --design-arch random \
    --critic-arch cat \
    --design-transform iid \
    --device <DEVICE>
```

To train a critic with equal interval designs, which is then used to evaluate the equal interval baseline, run the following
```bash
python3 epidemic.py \
    --num-steps 100000 \
    --num-experiments 5 \
    --lr 0.001 \
    --hidden-dim 512 \
    --encoding-dim 32 \
    --design-arch equal_interval \
    --critic-arch cat \
    --design-transform iid \
    --device <DEVICE>
```


Finally, to evaluate the different methods, run
```bash
python3 eval_epidemic.py \
    --experiment-id <ID> \
    --device <DEVICE>
```
