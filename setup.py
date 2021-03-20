from setuptools import setup, find_packages
from codecs import open
from rat import __version__

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('test-requirements.txt') as f:
    test_requirements = f.read().splitlines()

setup(
    name='rat',
    version=__version__,
    description='Relation-Aware Transformer for Portfolio Policy Learning',
    url='',
    author='Louis',
    author_email='louis.outin@gmail.com',
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=requirements,
    tests_require=test_requirements,
    entry_points={
        'console_scripts': [
            'rat=rat.main:run_main',
        ],
    }
)
