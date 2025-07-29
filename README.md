# Combining BioGeoChemical-Argo (BGC-Argo) floats and satellite observations for water column estimations of the particulate backscattering coefficient

This repository contains the code, experiments, and results for estimating vertical profiles of the particulate backscattering coefficient (bbp) in the upper 50 and 250 meters of the ocean. The models integrate in situ measurements from BGC-Argo profiling floats with satellite-derived reflectances and inherent optical properties (IOPs) retrieved from the Sentinel-3 OLCI sensor. The main objective is to evaluate how satellite-derived sea surface optical properties, when combined with BGC-Argo float measurements, can be used to estimate the vertical distribution of the bbp in the upper ocean.

## Related Publication

The study is described in detail in:

> **Combining BGC-Argo floats and satellite observations for water column estimations of the particulate backscattering coefficient**  
> Jorge García-Jiménez, Ana B. Ruescas, Julia Amorós-López, Raphaëlle Sauzède  
> *EGUsphere (2025)*  
> [🔗 Read the article](TODO)

## Regions of Interest

- North Atlantic
- Subtropical Gyres

<div align="center">
  <img src="docs/img/Map.png" alt="Map" width="90%"/>
</div>

## Modeling Approach

- **Model type**: Multi-output Random Forest Regressor
- **Input features**:
  - Satellite: OLCI reflectances (12 bands), C2RCC IOPs (8 parameters)
  - GlocColour + GlobalOcean: reflectances + Sea Level Anomaly (SLA)
  - BGC-Argo Float profiles: temperature, salinity, density, spiciness, Mixed Layer Depth (MLD)
  - Spatial-temporal: latitude, longitude, day of year
- **Target**: Particulate Backscattering Coefficient (Bbp) at 26 or 126 depth levels (0–50 m, 0–250 m)
  
## Model Performance – Deep Profiles (0–250 m)

<div align="center">
  <img src="docs/img/250.jpg" alt="Model Performance 250m" width="90%"/>
</div>

## Structure

<pre>
SatArgoBbp/
├── src/              # Modeling and utility functions
├── datasets/         # Processed input datasets (excluded from Git)
├── notebooks/        # Experimentation and validation notebooks
├── results/          # Outputs: plots, metrics, figures
├── docs/
│   └── img/          # Visuals for README and manuscript
├── .gitignore        # Ignore large/raw or intermediate files
├── README.md         # 
└── requirements.txt  # Python dependencies (optional)
</pre>

## How to Cite

If you use this repository, please cite our paper:

### APA

> García-Jiménez, J., Ruescas, A. B., Amorós-López, J., & Sauzède, R. (2025). *Combining BGC-Argo floats and satellite observations for water column estimations of the particulate backscattering coefficient*. EGUsphere. https://doi.org/10.5194/egusphere-2024-3942

### BibTeX

```bibtex
@article{garcia-jimenez2025bbp,
  author    = {García-Jiménez, Jorge and Ruescas, Ana B. and Amorós-López, Julia and Sauzède, Raphaëlle},
  title     = {Combining BGC-Argo floats and satellite observations for water column estimations of the particulate backscattering coefficient},
  journal   = {EGUsphere},
  year      = {2025},
  doi       = {10.5194/egusphere-2024-3942}
}
