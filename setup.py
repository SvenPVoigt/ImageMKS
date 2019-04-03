from distutils.core import setup

setup(
    name='ImageMKS',
    version='0.1.0dev',
    packages=['imagemks',],
    license='The MIT License (MIT)',
    long_description=open('README.md').read(),
    author='Sven Voigt',
    author_email='svenpvoigt@gmail.com',
    url='http://pypi.python.org/pypi/ImageMKS/',
    description='Sharing segmentation frameworks.',
    install_requires=[
        "torch >= 1.0.1",
        "scikit-image >= 0.14.1",
        "SciPy >= 1.1.0"
    ],
)
