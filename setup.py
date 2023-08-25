from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='Robustness-Framework',
    version='0.0.3',
    author='Mohammad KHalooei',
    url='https://github.com/khalooei/robustness-framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
