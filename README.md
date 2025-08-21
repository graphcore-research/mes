# Max Value Entropy Search Approximations

Building on this ICML 2025 spotlight paper: [A Unified Framework for Entropy Search and
Expected Improvement in Bayesian Optimization](https://arxiv.org/pdf/2501.18756)

Collections of toy expertiments for approximating Max Value entropy Search

![Distribution of Max Values gitven 3 points](scripts/original_data.png)

![Max value varying for one synthetic point](scripts/animation.gif)


# Setup
```
# install pytorch first
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# install the other dependencies
pip install -e .
```

# Usage
This makes the images and animation
```
cd scripts
python main.py
```