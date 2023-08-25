from setuptools import setup, find_packages

setup(
    name='Robustness-Framework',
    version='0.1.0',
    author='Mohammad KHalooei',
    author_email='mkhalooei90@gmail.com',
    description='An efficient framework for establishing a baseline for standard and adversarial machine learning training projects ',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)