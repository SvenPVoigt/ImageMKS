from distutils.core import setup
from setuptools import find_packages

setup(
    name='ImageMKS',
    version='0.0.1',
    packages=find_packages(),
    license='The MIT License (MIT)',
    long_description=open('README.md').read(),
    author='Sven Voigt',
    author_email='svenpvoigt@gmail.com',
    url='http://pypi.python.org/pypi/ImageMKS/',
    description='Sharing segmentation frameworks.',
    scripts=['bin/imagemks', 'bin/cellanalysis'],
    install_requires=[],
)
