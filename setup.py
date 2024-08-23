from setuptools import find_packages, setup

setup(
    name='BCItoolkit',
    version='0.1.0',
    description='A compiled and accessible brain-computer interface toolkit for those with minimal python experience',
    author='Max Mascini',    
    author_email="Mascini.Max@gmail.com",
    packages=find_packages(),
    install_requires=[],
    
    # setup_requires=['pytest-runner'],
    # tests_require=['pytest==4.4.1'],
    # test_suite='tests',
)