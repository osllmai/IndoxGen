from setuptools import setup, find_packages

# Read the requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Extract package names
packages = [req.split('==')[0] for req in requirements]

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='indoxGen',
    version='0.0.6',
    license='AGPL-3.0-or-later',
    packages=find_packages(),
    include_package_data=True,
    description='Indox Synthetic Data Generation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='nerdstudio',
    author_email='ashkan@nematifamilyfundation.onmicrosoft.com',
    url='https://github.com/osllmai/IndoxGen',
    keywords=[
        'AI',
        'deep learning',
        'language models',
        'synthetic data generation',
        'machine learning',
        'NLP'
    ],
    install_requires=[
        'loguru==0.7.2',
        'numpy==2.1.1',
        'pandas==2.2.3',
        'protobuf==3.20.3',
        'requests==2.32.3',
        'scikit_learn==1.4.2',
        'setuptools==69.5.1',
        'tenacity==8.2.2',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.9',
)
