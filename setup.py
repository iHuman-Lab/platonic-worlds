from setuptools import find_packages, setup

# ruff: noqa
setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A constraint-space representation learning which can provide a modality-independent description of the world, enabling heterogeneous observations to be mapped into a shared representation that preserves physical constraints and their temporal semantics.',
    author='Your name (or your organization/company/team)',
    license='MIT',
)
