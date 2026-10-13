from setuptools import setup, find_packages

setup(
    name="jax hardware co-simulation tool",
    version="0.1.0",
    author="Ishan Renjen, Ruhan Syed",
    description="Integration testing tool for ML hardware into JAX",
    packages=find_packages(),
    python_requires=">=3.12.15",
    install_requires=[],
)
