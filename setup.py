import pathlib

from setuptools import find_packages, setup

# The text of the README file
README_CONTENT = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name='torch-kerosene',
    version='0.2.2',
    description='Deep Learning framework for fast and clean research development with Pytorch',
    long_description=README_CONTENT,
    long_description_content_type='text/markdown',
    author='Benoit Anctil-Robitaille',
    author_email='benoit.anctil-robitaille.1@ens.etsmtl.ca',
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7"],
    packages=find_packages(exclude=("tests",)),
    install_requires=['numpy>=1.16.1',
                      'visdom>=0.1.8.8',
                      'pytorch-ignite>= 0.2.0',
                      'torch>=1.1',
                      'torchvision>=0.2.1',
                      'PyYAML']
)
