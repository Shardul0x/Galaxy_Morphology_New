# 🌌 Astrophysics Data Augmentation Pipeline

**VAE + PINN Framework for Physics-Informed Data Augmentation in Astrophysics Research**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A generic, physics-constrained simulation pipeline for generating synthetic astrophysics data using Variational Autoencoders (VAE) and Physics-Informed Neural Networks (PINN).

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Physics Domains](#physics-domains)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## ✨ Features

### Core Capabilities
- **Generic Framework**: Supports multiple astrophysics domains (galaxies, stars, exoplanets, orbits, cosmology)
- **Physics-Informed**: Enforces physical constraints (conservation laws, scaling relations, evolution equations)
- **Temporal Evolution**: Simulates time-dependent dynamics using PINN
- **Data Augmentation**: Generates synthetic data for ML training with validated physics
- **Multi-Format Support**: Load/export CSV, FITS, HDF5, NumPy, VOTable

### Physics Constraints
- **Stellar Physics**: Mass-luminosity relation, Stefan-Boltzmann law, HR diagram, main sequence lifetime
- **Orbital Mechanics**: Kepler's laws, energy/momentum conservation, vis-viva equation
- **Cosmology**: Hubble law, distance measures, Friedmann equations
- **Galaxy Dynamics**: Tully-Fisher relation, rotation curves, morphology consistency

### Visualization & Export
- Latent space visualization (2D/3D)
- Temporal evolution plots
- Training history and metrics
- Export to standard astronomy formats
- Interactive Streamlit web interface

## 🚀 Installation

### Prerequisites
Python >= 3.8
PyTorch >= 2.0
CUDA (optional, for GPU acceleration)

text

### Install from source
Clone the repository
git clone https://github.com/Shardul0x/ML_proj.git
cd ML_proj

Install dependencies
pip install -r requirements.txt

Install the package
pip install -e .

text

### Install specific components
For astronomy file formats (FITS, VOTable)
pip install astropy

For HDF5 support
pip install h5py tables

For interactive web interface
pip install streamlit

text

## 🎯 Quick Start

### 1. Load Data
from data.loader import AstrophysicsDataLoader

Load Galaxy Zoo data
loader = AstrophysicsDataLoader(
feature_columns=['P_EL', 'P_CW', 'P_ACW', 'P_EDGE',
'P_DK', 'P_MG', 'P_CS',
'P_EL_DEBIASED', 'P_CS_DEBIASED']
)
loader.load('GalaxyZoo1_DR_table2.csv')
X_train, X_test, _, _ = loader.split_train_test()

text

### 2. Preprocess
from data.preprocessor import AstrophysicsPreprocessor

preprocessor = AstrophysicsPreprocessor(
scaler_type='minmax',
clip_outliers=True
)
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

text

### 3. Configure Physics Domain
from config.physics_config import get_physics_domain

physics_domain = get_physics_domain('galaxy_morphology')
print(f"Features: {physics_domain.feature_names}")
print(f"Constraints: {[c.name for c in physics_domain.constraints]}")

text

### 4. Create & Train Models
import torch
from models.vae import GenericVAE
from models.pinn import GenericPINN
from train import VAETrainer, PINNTrainer
from config.model_config import VAEConfig, PINNConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VAE configuration
vae_config = VAEConfig(
input_dim=9,
hidden_dims=,
latent_dim=8,
num_epochs=50
)

vae = GenericVAE(
input_dim=vae_config.input_dim,
hidden_dims=vae_config.hidden_dims,
latent_dim=vae_config.latent_dim
)

Train VAE
trainer = VAETrainer(vae, vae_config, device)
history = trainer.train(train_loader, val_loader)

text

### 5. Augment Data
from data.augmentation import AstrophysicsAugmentor

augmentor = AstrophysicsAugmentor(vae, pinn, physics_domain, device)

augmented_data, evolutions = augmentor.augment_dataset(
X_train_scaled,
augmentation_factor=3,
include_evolution=True,
evolution_steps=30
)

print(f"Generated {len(augmented_data)} samples with {len(evolutions)} timesteps")

text

### 6. Visualize & Export
from visualization.plotter import AstrophysicsVisualizer
from visualization.export import AstrophysicsExporter

Visualize latent space
visualizer = AstrophysicsVisualizer()
visualizer.plot_latent_space_2d(latent_vectors, save_path='latent_space.png')

Export results
exporter = AstrophysicsExporter()
exporter.export_to_fits(
augmented_data,
'augmented_catalog.fits',
column_names=physics_domain.feature_names
)

text

## 🌟 Physics Domains

### 1. Galaxy Morphology
physics_domain = get_physics_domain('galaxy_morphology')

text
- **Features**: Ellipticity, spiral arms (CW/ACW), edge-on probability, dust, merger, concentration
- **Constraints**: Morphology consistency, bulge-disc relations
- **Use Cases**: Galaxy classification, morphological evolution

### 2. Stellar Evolution
physics_domain = get_physics_domain('stellar_evolution')

text
- **Features**: Mass, luminosity, temperature, radius, metallicity, age
- **Constraints**: Mass-luminosity relation, Stefan-Boltzmann law, HR diagram, main sequence lifetime
- **Use Cases**: Stellar parameter estimation, age prediction

### 3. Exoplanet Transits
physics_domain = get_physics_domain('exoplanet_transits')

text
- **Features**: Period, depth, duration, impact parameter, planet/stellar radius
- **Constraints**: Kepler's third law, transit geometry
- **Use Cases**: Transit detection, parameter characterization

### 4. Orbital Dynamics
physics_domain = get_physics_domain('orbital_dynamics')

text
- **Features**: Semi-major axis, eccentricity, inclination, orbital angles
- **Constraints**: Kepler's laws, energy/momentum conservation
- **Use Cases**: Orbit prediction, stability analysis

### 5. Cosmological Parameters
physics_domain = get_physics_domain('cosmological_parameters')

text
- **Features**: Redshift, distances (angular/luminosity/comoving), age, lookback time
- **Constraints**: Hubble law, distance relations, Friedmann equations
- **Use Cases**: Cosmological simulations, distance ladder

## 🏗️ Architecture

astrophysics_augmentation/
├── config/
│ ├── physics_config.py # Physics domain configurations
│ └── model_config.py # Model hyperparameters
├── models/
│ ├── vae.py # Generic VAE implementation
│ ├── pinn.py # Physics-informed neural network
│ └── losses.py # Loss functions
├── physics/
│ ├── base.py # Base physics classes
│ ├── stellar.py # Stellar physics
│ ├── orbital.py # Orbital mechanics
│ ├── cosmological.py # Cosmology
│ └── galaxy.py # Galaxy dynamics
├── data/
│ ├── loader.py # Multi-format data loading
│ ├── preprocessor.py # Data preprocessing
│ └── augmentation.py # Augmentation pipeline
├── visualization/
│ ├── plotter.py # Scientific visualization
│ └── export.py # Data export utilities
├── examples/
│ ├── galaxy_morphology_example.py
│ ├── stellar_evolution_example.py
│ └── exoplanet_transits_example.py
├── train.py # Training pipeline
├── app.py # Streamlit web interface
└── README.md

text

## 📚 Usage Examples

See `examples/` directory for complete working examples:

- **Galaxy Morphology**: `examples/galaxy_morphology_example.py`
- **Stellar Evolution**: `examples/stellar_evolution_example.py`
- **Exoplanet Transits**: `examples/exoplanet_transits_example.py`

### Run Examples
python examples/galaxy_morphology_example.py
python examples/stellar_evolution_example.py
python examples/exoplanet_transits_example.py

text

### Launch Web Interface
streamlit run app.py

text

## 📖 API Reference

[Complete API documentation will be generated using Sphinx]

### Key Classes

**Models**
- `GenericVAE`: Flexible VAE architecture
- `GenericPINN`: Physics-informed neural network
- `VAETrainer`: Training pipeline for VAE
- `PINNTrainer`: Training pipeline for PINN

**Data**
- `AstrophysicsDataLoader`: Multi-format data loading
- `AstrophysicsPreprocessor`: Data preprocessing
- `AstrophysicsAugmentor`: Data augmentation pipeline

**Physics**
- `PhysicsEquation`: Base class for physics constraints
- `PowerLaw`, `LinearRelation`: Generic physics relations
- `ConservationLaw`: Conservation constraints

**Visualization**
- `AstrophysicsVisualizer`: Scientific plotting
- `AstrophysicsExporter`: Export to astronomy formats

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding New Physics Domains
1. Define physics constraints in `physics/` module
2. Create domain configuration in `config/physics_config.py`
3. Add example script in `examples/`
4. Update documentation

## 📄 Citation

If you use this pipeline in your research, please cite:

@software{astrophysics_augmentation_2025,
title={Astrophysics Data Augmentation Pipeline: VAE + PINN Framework},
author={Your Name},
year={2025},
url={https://github.com/Shardul0x/ML_proj}
}

text

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Galaxy Zoo dataset for morphology features
- PyTorch team for deep learning framework
- Astropy community for astronomy tools

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [https://github.com/Shardul0x/ML_proj/issues](https://github.com/Shardul0x/ML_proj/issues)
- Email: your.email@example.com

---

**Made with ❤️ for Astrophysics Research**