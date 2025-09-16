# ACE

**ACE (Aging Cell Embedding)** is an explainable deep generative model for **disentangling aging-related signals** from background biological variation in single-cell transcriptomic data.  

ACE builds two separate latent spaces:
- **Aging latent space** – captures gene expression patterns related to aging  
- **Background latent space** – models confounding factors such as tissue, cell type, or species differences  

This enables ACE to identify both **global aging markers** (shared across tissues and cell types) and **local, tissue- or cell-type-specific aging signals**, and supports **cross-species alignment** of aging trajectories.  
ACE is implemented on top of the [scvi-tools](https://scvi-tools.org/) framework.

---

## Installation

ACE is **not yet published to PyPI**. You can install it locally for development and testing.

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/ace.git
   cd ace
   ```
2. Create and activate the Conda environment
   ```bash
   conda env create -f environment.yml
   conda activate ace
   ```
3. Install in editable mode
   ```bash
   pip install -e .
   ```
