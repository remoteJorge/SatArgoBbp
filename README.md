# Combining BioGeoChemical-Argo (BGC-Argo) floats and satellite observations for water column estimations of the particulate backscattering coefficient

This repository contains the code, experiments, and results for estimating vertical profiles of the particulate backscattering coefficient (bbp) in the upper 50 and 250 meters of the ocean. The models integrate in situ measurements from BGC-Argo profiling floats with satellite-derived reflectances and inherent optical properties (IOPs) retrieved from the Sentinel-3 OLCI sensor. The main objective is to evaluate how satellite-derived sea surface optical properties, when combined with BGC-Argo float measurements, can be used to estimate the vertical distribution of the bbp in the upper ocean.

## Related Publication

The study is described in detail in the following paper:

> **Combining BGC-Argo floats and satellite observations for water column estimations of the particulate backscattering coefficient**  
> Jorge García-Jiménez, Ana B. Ruescas, Julia Amorós-López, Raphaëlle Sauzède  
> *EGUsphere (2025)*  
> [🔗 Read the article](TODO)

## Structure

- `src/`: Modeling and utility functions
- `datasets/`: Processed input datasets
- `notebooks/`: Experimentation and validation
- `results/`: Outputs like plots, metrics, and figures

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
