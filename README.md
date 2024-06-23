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
