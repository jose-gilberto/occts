""" Setup module to install the package locally.
"""
import os
from setuptools import setup


def read_file_content(filename: str) -> str:
    return open(
        os.path.join(os.path.dirname(__file__), filename),
        encoding='utf-8'
    ).read()


setup(
    name='occts',
    version='0.1.0',
    author='Gilberto Medeiros',
    author_email='medeiros.gilberto.br@gmail.com',
    description=('Library with methods and algorithms to do '
                    'one class classification with time series.'),
    license='MIT',
    keywords='time series one-class classification deep learning',
    # url='https://packages.python.org',
    packages=['occts'],
    long_description=read_file_content('README.md'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License'
    ]
)