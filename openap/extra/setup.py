from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyBada',
    version='1.1',
    description='Base of Aircraft Data (BADA) performance and optimisation',
    long_description=long_description,
    url='https://gitlab.com/ramondalmau/pyBada.git',
    author='Ramon Dalmau',
    author_email='ramon.dalmau@upc.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='BADA performance optimisation',
    packages=["pyBada"],
    install_requires=['casadi','datetime']
)
