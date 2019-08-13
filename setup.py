import pathlib

from setuptools import find_packages, setup

# The text of the README file
README_CONTENT = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name='kerosene',
    version='0.0.1',
    description='Pytorch Framework For Medical Image Analysis',
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
    install_requires=['matplotlib',
                      'PyYAML>=5.1',
                      'PyHamcrest>=1.9.0',
                      'nibabel>=2.3.3',
                      'nilearn==0.5.0',
                      'pynrrd>=0.4.0',
                      'numba==0.42.1',
                      'numpy==1.16.1',
                      'pandas==0.24.1',
                      'pyparsing==2.3.1',
                      'pytest==4.3.0',
                      'scikit-learn>=0.20.2',
                      'scipy==1.2.1',
                      'torch>=1.1',
                      'torchfile==0.1.0',
                      'torchvision>=0.2.1',
                      'visdom==0.1.8.8',
                      'imbalanced-learn',
                      'blinker>=1.4',
                      'pytorch-ignite>= 0.2.0']
)
