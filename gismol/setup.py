
from setuptools import setup, find_packages

setup(
    name='gismol',
    version='0.1.0',
    description='General Intelligent Systems Modelling Language (COH 9‑tuple implementation)',
    author='Harris Wang',
    author_email='harrisw@athabascau.ca',
    url='https://github.com/harriswang/gismol',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'networkx>=2.6',
        'torch>=1.10',
        'matplotlib>=3.4',
    ],
    extras_require={
        'dev': ['pytest', 'sphinx', 'black']
    },
    python_requires='>=3.9',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
