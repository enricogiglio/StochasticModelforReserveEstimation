# StochasticModelforReserveEstimation
This repository contains a comprehensive methodology for stochastic reserve estimation in power systems, applied to South Africa's 2050 grid. The Python code simulates reserve requirements, considering tripping events, load uncertainties, renewable energy variability, and ramp dynamincs following ENTSO-E guidelines.

## Overview

The repository includes Python scripts for:
- Generating sawtooth sequences to simulate ramping dynamics.
- Evaluating tripping events and their impact on reserve requirements.
- Evaluating frequency imbalance due to unpredictable load variations and renewable generation.
- Calculating reserve needs (FCR, aFRR, mFRR, RR) using a stochastic approach.
- Building reserve allocation strategies for different network configurations.

## Repository Structure

- `reserve_estimation.py`: Main script containing functions for reserve estimation.
- `data/`: Directory containing input data files, such as network configurations and scenario parameters.

## Getting Started

### Prerequisites
- Required Python packages: pypsa, numpy, pandas, sklearn, matplotlib

### Usage
- Place your network configuration file (e.g., network_SA2050.csv) in the data/ directory.
- Place the combinations of components of components for which you want to estimate the reserve requirements file (e.g., CombToStudy.xlsx) in the data/ directory.
- Update the file_path and network_name variables in reserve_estimation.py with your input files.
- Run the main script
```
 python reserve_estimation.py
```

### Functions
Each function in the script includes detailed docstrings for understanding its purpose and usage. Key functions include:

- `generate_saw_sequence()`: Generates a sawtooth sequence for the given time horizon and variation of residual demand (ramp dynamics).
- `tripping()`: Evaluates tripping events for network components over the given years.
- `reserve_dimensioning()`: Calculates reserve requirements based on specified quantiles using a stochastic approach.
- `reserve_function_builder()`: Main function that activates all the other sub-functions.

# Collaboration
These codes were developed in collaboration with @davide-f



# Citation

If you use this work, please cite:

```bibtex
@article{GIGLIO2025,
  title = {Integrated stochastic reserve estimation and MILP energy planning for high renewable penetration: Application to 2050 South African energy system},
  journal = {Sustainable Energy, Grids and Networks},
  volume = {42},
  pages = {101650},
  year = {2025},
  issn = {2352-4677},
  doi = {https://doi.org/10.1016/j.segan.2025.101650},
  url = {https://www.sciencedirect.com/science/article/pii/S2352467725000323},
  author = {Enrico Giglio and Davide Fioriti and Munyaradzi Justice Chihota and Davide Poli and Bernard Bekker and Giuliana Mattiazzo}
}

