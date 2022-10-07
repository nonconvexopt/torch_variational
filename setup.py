from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup_requires = [
    ]

install_requires = [
    'torch>=1.6.0',
    ]

setup(
    name='torch_variational',
    version='0.1',
    description='Variational Modules for PyTorch',
    author='nonconvexopt',
    author_email='nonconvexopt@gmail.com',
    packages=find_packages(),
    install_requires=install_requires,
    package_dir={'': '.'},
)