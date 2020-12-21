# Setup
## Preliminaries
The experiments have been performed using `python 3.7` and this hardware:
* AMD Ryzen 7 2700X Eight-Core Processor
* 34 GB RAM
* GeForce RTX 2080 Ti

### Setup python environment.
## Using Conda
```bash
conda create --name dpg_gnn python=3.7
conda install -y -q --name dpg_gnn -c conda-forge --file requirements.txt
conda activate dpg_gnn
```

## Using Virtualenv + pip
```bash
virtualenv dpg_gnn -p `which python3.7`
source dpg_gnn/bin/activate
pip install -r requirements.txt
```
