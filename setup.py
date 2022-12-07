from setuptools import setup, find_packages

setup(
    name='emma',
    maintainer='zbenmo@gmail.com',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.5.2, <1.6',
        'numpy>=1.23.5, <1.24',
    ],
    extras_require={
        'examples': [
            'scikit-learn>=1.1.3, < 1.2'
        ],
    },
)