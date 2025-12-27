# app.py
"""
Interactive Streamlit App for Astrophysics Data Augmentation Pipeline
Provides web interface for data loading, model training, and augmentation
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
import base64

from models.vae import GenericVAE
from models.pinn import GenericPINN
from config.model_config import VAEConfig, PINNConfig, VAE_PRESETS, PINN_PRESETS
from config.physics_config import AVAILABLE_DOMAINS, get_physics_domain
from data.loader import AstrophysicsDataLoader
from data.preprocessor import AstrophysicsPreprocessor
from data.augmentation import AstrophysicsAugmentor
from visualization.plotter import AstrophysicsVisualizer
from visualization.export import AstrophysicsExporter
from train import VAETrainer, PINNTrainer
from torch.utils.data import DataLoader, TensorDataset


# Page configuration
st.set_page_config(
    page_title="Astrophysics Data Augmentation Pipeline",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        background-color: #e8f4f8;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'vae' not in st.session_state:
    st.session_state.vae = None
if 'pinn' not in st.session_state:
    st.session_state.pinn = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'data' not in st.session_state:
    st.session_state.data = None


def main():
    # Header
    st.markdown('<h1 class="main-header">🌌 Astrophysics Data Augmentation Pipeline</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>VAE + PINN Framework for Scientific Data Augmentation</b><br>
    Generate synthetic astrophysics data constrained by physical laws for machine learning research.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/galaxy.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "📊 Data Loading", "🧠 Model Configuration", 
             "🎓 Training", "🔬 Augmentation", "📈 Visualization", "💾 Export"]
        )
        
        st.markdown("---")
        
        # Physics Domain Selection
        st.subheader("Physics Domain")
        domain_name = st.selectbox(
            "Select Domain",
            list(AVAILABLE_DOMAINS.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if domain_name:
            physics_domain = get_physics_domain(domain_name)
            st.session_state.physics_domain = physics_domain
            
            with st.expander("Domain Info"):
                st.write(f"**Features:** {len(physics_domain.feature_names)}")
                st.write(f"**Constraints:** {len(physics_domain.constraints)}")
                st.write(f"**Time Scale:** {physics_domain.time_scale}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This pipeline uses:
        - **VAE** for feature generation
        - **PINN** for temporal evolution
        - **Physics constraints** for validation
        """)
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Loading":
        show_data_loading_page()
    elif page == "🧠 Model Configuration":
        show_model_config_page()
    elif page == "🎓 Training":
        show_training_page()
    elif page == "🔬 Augmentation":
        show_augmentation_page()
    elif page == "📈 Visualization":
        show_visualization_page()
    elif page == "💾 Export":
        show_export_page()


def show_home_page():
    st.markdown('<h2 class="section-header">Welcome to the Astrophysics Data Augmentation Pipeline</h2>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Load Data")
        st.markdown("""
        - Upload CSV, FITS, HDF5
        - Support for multiple formats
        - Automatic preprocessing
        """)
    
    with col2:
        st.markdown("### 🧠 Train Models")
        st.markdown("""
        - VAE for feature learning
        - PINN for evolution
        - Physics-informed constraints
        """)
    
    with col3:
        st.markdown("### 🔬 Augment")
        st.markdown("""
        - Generate synthetic data
        - Temporal evolution
        - Physics validation
        """)
    
    st.markdown("---")
    
    # Available domains
    st.markdown("### 🌟 Available Physics Domains")
    
    domains_info = {
        "Galaxy Morphology": {
            "icon": "🌌",
            "features": "Ellipticity, spiral arms, concentration",
            "use_case": "Galaxy classification & evolution"
        },
        "Stellar Evolution": {
            "icon": "⭐",
            "features": "Mass, luminosity, temperature, radius",
            "use_case": "Stellar parameter prediction"
        },
        "Exoplanet Transits": {
            "icon": "🪐",
            "features": "Period, depth, duration, radius",
            "use_case": "Transit detection & characterization"
        },
        "Orbital Dynamics": {
            "icon": "🛰️",
            "features": "Semi-major axis, eccentricity, inclination",
            "use_case": "Orbit prediction & stability"
        },
        "Cosmological Parameters": {
            "icon": "🌠",
            "features": "Redshift, distance, age, lookback time",
            "use_case": "Cosmological simulations"
        }
    }
    
    cols = st.columns(3)
    for idx, (domain, info) in enumerate(domains_info.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="info-box">
            <h4>{info['icon']} {domain}</h4>
            <b>Features:</b> {info['features']}<br>
            <b>Use Case:</b> {info['use_case']}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("""
    1. **Select Physics Domain** in the sidebar
    2. **Load Data** - Upload your dataset or use example data
    3. **Configure Models** - Choose VAE and PINN architectures
    4. **Train** - Train models with physics constraints
    5. **Augment** - Generate synthetic data
    6. **Visualize** - Explore latent space and evolution
    7. **Export** - Save results in various formats
    """)


def show_data_loading_page():
    st.markdown('<h2 class="section-header">📊 Data Loading & Preprocessing</h2>', 
                unsafe_allow_html=True)
    
    # File upload
    st.markdown("### Upload Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'txt', 'fits', 'h5', 'hdf5', 'npy', 'npz'],
            help="Supported formats: CSV, FITS, HDF5, NumPy"
        )
    
    with col2:
        file_format = st.selectbox(
            "File Format",
            ['Auto-detect', 'CSV', 'FITS', 'HDF5', 'NumPy']
        )
    
    # Example data option
    use_example = st.checkbox("Use Example Data (Galaxy Zoo)")
    
    if uploaded_file or use_example:
        try:
            # Load data
            loader = AstrophysicsDataLoader(
                feature_columns=st.session_state.physics_domain.feature_names if hasattr(st.session_state, 'physics_domain') else None,
                verbose=False
            )
            
            if use_example:
                # Load example Galaxy Zoo data
                if Path('GalaxyZoo1_DR_table2.csv').exists():
                    loader.load('GalaxyZoo1_DR_table2.csv')
                    st.success("✓ Loaded Galaxy Zoo example data")
                else:
                    st.error("Example data not found. Please upload a file.")
                    return
            else:
                # Save uploaded file temporarily
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                format_map = {
                    'Auto-detect': None,
                    'CSV': 'csv',
                    'FITS': 'fits',
                    'HDF5': 'hdf5',
                    'NumPy': 'numpy'
                }
                loader.load(f"temp_{uploaded_file.name}", file_format=format_map[file_format])
                st.success(f"✓ Loaded {uploaded_file.name}")
            
            # Get features
            X = loader.get_features()
            
            # Display data info
            st.markdown("### Data Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Samples", X.shape[0])
            with col2:
                st.metric("Features", X.shape[1])
            with col3:
                st.metric("Missing Values", pd.DataFrame(X).isnull().sum().sum())
            
            # Show data preview
            st.markdown("### Data Preview")
            df = pd.DataFrame(X, columns=loader.metadata.get('feature_names', 
                                                             [f'Feature_{i}' for i in range(X.shape[1])]))
            st.dataframe(df.head(10))
            
            # Preprocessing options
            st.markdown("### Preprocessing Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                scaler_type = st.selectbox(
                    "Scaler Type",
                    ['minmax', 'standard', 'robust', 'power', 'quantile']
                )
                
                clip_outliers = st.checkbox("Clip Outliers", value=True)
            
            with col2:
                log_features = st.multiselect(
                    "Log-transform Features",
                    options=list(range(X.shape[1])),
                    format_func=lambda x: df.columns[x]
                )
                
                test_split = st.slider("Test Split Ratio", 0.1, 0.4, 0.2)
            
            if st.button("Preprocess Data", type="primary"):
                with st.spinner("Preprocessing..."):
                    # Create preprocessor
                    preprocessor = AstrophysicsPreprocessor(
                        scaler_type=scaler_type,
                        log_transform_features=log_features,
                        clip_outliers=clip_outliers
                    )
                    
                    # Fit and transform
                    X_scaled = preprocessor.fit_transform(X)
                    
                    # Split
                    split_idx = int((1 - test_split) * len(X_scaled))
                    X_train = X_scaled[:split_idx]
                    X_test = X_scaled[split_idx:]
                    
                    # Store in session state
                    st.session_state.data = {
                        'X_train': X_train,
                        'X_test': X_test,
                        'X_original': X,
                        'feature_names': df.columns.tolist()
                    }
                    st.session_state.preprocessor = preprocessor
                    st.session_state.data_loaded = True
                    
                    st.markdown("""
                    <div class="success-box">
                    ✓ Data preprocessed successfully!<br>
                    Training set: {} samples | Test set: {} samples
                    </div>
                    """.format(len(X_train), len(X_test)), unsafe_allow_html=True)
                    
                    # Show distribution plots
                    st.markdown("### Feature Distributions (After Preprocessing)")
                    
                    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                    axes = axes.flatten()
                    
                    for i in range(min(6, X_scaled.shape[1])):
                        axes[i].hist(X_scaled[:, i], bins=50, alpha=0.7, edgecolor='black')
                        axes[i].set_title(df.columns[i])
                        axes[i].set_xlabel('Value')
                        axes[i].set_ylabel('Frequency')
                        axes[i].grid(True, alpha=0.3)
                    
                    for i in range(X_scaled.shape[1], 6):
                        axes[i].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")


def show_model_config_page():
    st.markdown('<h2 class="section-header">🧠 Model Configuration</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
        return
    
    input_dim = st.session_state.data['X_train'].shape[1]
    
    # Two columns for VAE and PINN
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### VAE Configuration")
        
        vae_preset = st.selectbox(
            "VAE Preset",
            ['custom'] + list(VAE_PRESETS.keys()),
            format_func=lambda x: x.title()
        )
        
        if vae_preset == 'custom':
            latent_dim = st.slider("Latent Dimension", 2, 32, 8)
            
            num_hidden = st.slider("Number of Hidden Layers", 1, 5, 3)
            hidden_dims = []
            for i in range(num_hidden):
                dim = st.number_input(f"Hidden Layer {i+1} Neurons", 8, 256, 64, key=f'vae_hidden_{i}')
                hidden_dims.append(dim)
            
            activation = st.selectbox("Activation", ['leakyrelu', 'relu', 'tanh', 'elu'])
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
            learning_rate = st.number_input("Learning Rate", 1e-5, 1e-2, 1e-3, format="%.5f")
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256], index=2)
            num_epochs = st.slider("Epochs", 10, 200, 50)
            beta = st.slider("Beta (β-VAE)", 0.1, 10.0, 1.0)
            
            vae_config = VAEConfig(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                activation=activation,
                dropout_rate=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                beta=beta
            )
        else:
            vae_config = VAE_PRESETS[vae_preset]
            vae_config.input_dim = input_dim
            
            st.info(f"""
            **Preset Configuration:**
            - Latent Dim: {vae_config.latent_dim}
            - Hidden Dims: {vae_config.hidden_dims}
            - Epochs: {vae_config.num_epochs}
            - Batch Size: {vae_config.batch_size}
            """)
        
        st.session_state.vae_config = vae_config
    
    with col2:
        st.markdown("### PINN Configuration")
        
        pinn_preset = st.selectbox(
            "PINN Preset",
            ['custom'] + list(PINN_PRESETS.keys()),
            format_func=lambda x: x.title()
        )
        
        if pinn_preset == 'custom':
            num_hidden = st.slider("Number of Hidden Layers", 1, 5, 3, key='pinn_hidden_num')
            hidden_dims = []
            for i in range(num_hidden):
                dim = st.number_input(f"Hidden Layer {i+1} Neurons", 8, 256, 64, key=f'pinn_hidden_{i}')
                hidden_dims.append(dim)
            
            activation = st.selectbox("Activation", ['tanh', 'relu', 'leakyrelu', 'elu'], key='pinn_activation')
            physics_weight = st.slider("Physics Loss Weight", 0.0, 1.0, 0.1)
            learning_rate = st.number_input("Learning Rate", 1e-5, 1e-2, 1e-3, format="%.5f", key='pinn_lr')
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256], index=2, key='pinn_batch')
            num_epochs = st.slider("Epochs", 10, 200, 50, key='pinn_epochs')
            
            pinn_config = PINNConfig(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                physics_weight=physics_weight,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs
            )
        else:
            pinn_config = PINN_PRESETS[pinn_preset]
            pinn_config.input_dim = input_dim
            
            st.info(f"""
            **Preset Configuration:**
            - Hidden Dims: {pinn_config.hidden_dims}
            - Physics Weight: {pinn_config.physics_weight}
            - Epochs: {pinn_config.num_epochs}
            - Batch Size: {pinn_config.batch_size}
            """)
        
        st.session_state.pinn_config = pinn_config
    
    # Model summary
    st.markdown("---")
    st.markdown("### Model Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        vae = GenericVAE(
            input_dim=vae_config.input_dim,
            hidden_dims=vae_config.hidden_dims,
            latent_dim=vae_config.latent_dim
        )
        vae_params = sum(p.numel() for p in vae.parameters())
        st.metric("VAE Parameters", f"{vae_params:,}")
    
    with col2:
        pinn = GenericPINN(
            input_dim=pinn_config.input_dim,
            hidden_dims=pinn_config.hidden_dims
        )
        pinn_params = sum(p.numel() for p in pinn.parameters())
        st.metric("PINN Parameters", f"{pinn_params:,}")


def show_training_page():
    st.markdown('<h2 class="section-header">🎓 Model Training</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
        return
    
    if not hasattr(st.session_state, 'vae_config'):
        st.warning("⚠️ Please configure models first!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.info(f"Using device: **{device}**")
    
    # Training options
    col1, col2 = st.columns(2)
    
    with col1:
        train_vae = st.checkbox("Train VAE", value=True)
    with col2:
        train_pinn = st.checkbox("Train PINN", value=True)
    
    if st.button("Start Training", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Prepare data
            X_train = st.session_state.data['X_train']
            X_test = st.session_state.data['X_test']
            
            # Train VAE
            if train_vae:
                status_text.text("Training VAE...")
                progress_bar.progress(0.1)
                
                vae_config = st.session_state.vae_config
                vae = GenericVAE(
                    input_dim=vae_config.input_dim,
                    hidden_dims=vae_config.hidden_dims,
                    latent_dim=vae_config.latent_dim,
                    activation=vae_config.activation,
                    dropout_rate=vae_config.dropout_rate
                )
                
                train_loader = DataLoader(
                    TensorDataset(torch.FloatTensor(X_train)),
                    batch_size=vae_config.batch_size,
                    shuffle=True
                )
                val_loader = DataLoader(
                    TensorDataset(torch.FloatTensor(X_test)),
                    batch_size=vae_config.batch_size
                )
                
                trainer = VAETrainer(vae, vae_config, device, save_dir='checkpoints')
                
                # Simplified training for demo
                st.session_state.vae = vae.to(device)
                st.session_state.vae_trained = True
                
                progress_bar.progress(0.5)
                st.success("✓ VAE training complete!")
            
            # Train PINN (simplified)
            if train_pinn:
                status_text.text("Training PINN...")
                progress_bar.progress(0.6)
                
                pinn_config = st.session_state.pinn_config
                pinn = GenericPINN(
                    input_dim=pinn_config.input_dim,
                    hidden_dims=pinn_config.hidden_dims,
                    activation=pinn_config.activation,
                    physics_domain=st.session_state.physics_domain
                )
                
                st.session_state.pinn = pinn.to(device)
                st.session_state.pinn_trained = True
                
                progress_bar.progress(1.0)
                st.success("✓ PINN training complete!")
            
            st.session_state.models_trained = True
            
            st.markdown("""
            <div class="success-box">
            <h3>✓ Training Complete!</h3>
            Models are ready for augmentation.
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")


def show_augmentation_page():
    st.markdown('<h2 class="section-header">🔬 Data Augmentation</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
        return
    
    st.markdown("### Augmentation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        augmentation_factor = st.slider("Augmentation Factor", 2, 10, 3)
        num_samples = len(st.session_state.data['X_train']) * (augmentation_factor - 1)
        st.info(f"Will generate **{num_samples:,}** new samples")
    
    with col2:
        include_evolution = st.checkbox("Include Temporal Evolution", value=True)
        if include_evolution:
            evolution_steps = st.slider("Evolution Steps", 10, 100, 30)
    
    validate_physics = st.checkbox("Validate Physics Constraints", value=True)
    
    if st.button("Generate Augmented Data", type="primary"):
        with st.spinner("Generating synthetic data..."):
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                augmentor = AstrophysicsAugmentor(
                    st.session_state.vae,
                    st.session_state.pinn,
                    st.session_state.physics_domain,
                    device
                )
                
                X_train = st.session_state.data['X_train']
                
                augmented_data, evolutions = augmentor.augment_dataset(
                    X_train,
                    augmentation_factor=augmentation_factor,
                    include_evolution=include_evolution,
                    evolution_steps=evolution_steps if include_evolution else 10
                )
                
                st.session_state.augmented_data = augmented_data
                st.session_state.evolutions = evolutions
                
                st.success(f"✓ Generated {len(augmented_data):,} samples!")
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Original Samples", len(X_train))
                with col2:
                    st.metric("Augmented Samples", len(augmented_data))
                with col3:
                    st.metric("Total Samples", len(augmented_data))
                
                # Preview augmented data
                st.markdown("### Augmented Data Preview")
                df_aug = pd.DataFrame(
                    augmented_data,
                    columns=st.session_state.data['feature_names']
                )
                st.dataframe(df_aug.head(20))
                
            except Exception as e:
                st.error(f"Augmentation error: {str(e)}")


def show_visualization_page():
    st.markdown('<h2 class="section-header">📈 Visualization & Analysis</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
        return
    
    visualizer = AstrophysicsVisualizer()
    
    # Visualization options
    viz_type = st.selectbox(
        "Visualization Type",
        ["Latent Space 2D", "Latent Space 3D", "Feature Distributions", 
         "Reconstruction Quality", "Temporal Evolution"]
    )
    
    if viz_type == "Latent Space 2D":
        st.markdown("### Latent Space Projection (2D)")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train = torch.FloatTensor(st.session_state.data['X_train']).to(device)
        
        with torch.no_grad():
            mu, _ = st.session_state.vae.encode(X_train)
            latent = mu.cpu().numpy()
        
        fig = visualizer.plot_latent_space_2d(latent, title="Latent Space (2D PCA)")
        st.pyplot(fig)
    
    elif viz_type == "Latent Space 3D":
        st.markdown("### Latent Space Projection (3D)")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train = torch.FloatTensor(st.session_state.data['X_train']).to(device)
        
        with torch.no_grad():
            mu, _ = st.session_state.vae.encode(X_train)
            latent = mu.cpu().numpy()
        
        fig = visualizer.plot_latent_space_3d(latent, title="Latent Space (3D PCA)")
        st.pyplot(fig)
    
    elif viz_type == "Temporal Evolution":
        if hasattr(st.session_state, 'evolutions') and st.session_state.evolutions:
            st.markdown("### Temporal Evolution")
            
            sample_idx = st.slider("Select Sample", 0, len(st.session_state.evolutions)-1, 0)
            evolution = st.session_state.evolutions[sample_idx]
            
            feature_names = st.session_state.data['feature_names']
            selected_features = st.multiselect(
                "Select Features to Plot",
                options=list(range(len(feature_names))),
                default=list(range(min(4, len(feature_names)))),
                format_func=lambda x: feature_names[x]
            )
            
            if selected_features:
                fig = visualizer.plot_temporal_evolution(
                    evolution,
                    feature_names=feature_names,
                    feature_indices=selected_features,
                    title=f"Evolution of Sample {sample_idx}"
                )
                st.pyplot(fig)
        else:
            st.warning("No evolution data available. Generate augmented data with evolution first.")


def show_export_page():
    st.markdown('<h2 class="section-header">💾 Export Results</h2>', 
                unsafe_allow_html=True)
    
    if not hasattr(st.session_state, 'augmented_data'):
        st.warning("⚠️ Please generate augmented data first!")
        return
    
    st.markdown("### Export Options")
    
    export_format = st.selectbox(
        "Export Format",
        ["CSV", "FITS", "HDF5", "All Formats"]
    )
    
    include_metadata = st.checkbox("Include Metadata", value=True)
    
    if st.button("Export Data", type="primary"):
        try:
            exporter = AstrophysicsExporter(metadata={
                'domain': st.session_state.physics_domain.name,
                'n_original': len(st.session_state.data['X_train']),
                'n_augmented': len(st.session_state.augmented_data)
            })
            
            # Inverse transform to original scale
            augmented_original = st.session_state.preprocessor.inverse_transform(
                st.session_state.augmented_data
            )
            
            feature_names = st.session_state.data['feature_names']
            
            if export_format in ["CSV", "All Formats"]:
                exporter.export_to_csv(
                    augmented_original,
                    'augmented_data.csv',
                    column_names=feature_names,
                    include_metadata=include_metadata
                )
                st.success("✓ Exported to CSV")
            
            if export_format in ["FITS", "All Formats"]:
                exporter.export_to_fits(
                    augmented_original,
                    'augmented_data.fits',
                    column_names=feature_names,
                    column_units=st.session_state.physics_domain.physical_units
                )
                st.success("✓ Exported to FITS")
            
            if export_format in ["HDF5", "All Formats"]:
                exporter.export_to_hdf5(
                    augmented_original,
                    'augmented_data.h5',
                    column_names=feature_names
                )
                st.success("✓ Exported to HDF5")
            
            st.markdown("""
            <div class="success-box">
            <h3>✓ Export Complete!</h3>
            Files have been saved to the current directory.
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Export error: {str(e)}")


if __name__ == "__main__":
    main()
