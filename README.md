## 🧪 Environment Setup

To ensure a smooth and reproducible setup, please follow the steps below.

### 1️⃣ Create a Conda Environment
Create a dedicated Conda environment with Python 3.10:

```bash
conda create -n project-1 python=3.10.0

conda install -n project-1 cudatoolkit=11.2 cudnn=8.1 -c conda-forge

conda activate project-1
