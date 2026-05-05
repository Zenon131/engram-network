from setuptools import find_packages, setup

setup(
    name="bdh",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "numpy>=1.21.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "mlx": ["mlx"],  # Apple Silicon MLX support
    },
    author="BDH Team",
    description="Baby Dragon Hathcling (BDH) Model",
    python_requires=">=3.8",
)
