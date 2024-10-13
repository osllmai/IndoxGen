from setuptools import setup, find_packages

# Read the requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Extract package names
packages = [req.split('==')[0] for req in requirements if '==' in req]

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='indoxGen-tensor',
    version='0.0.3',
    license='AGPL-3.0-or-later',
    packages=find_packages(where='libs', include=['indoxGen_tensor', 'indoxGen_tensor.*']),
    package_dir={'': 'libs'},
    include_package_data=True,
    description='Indox Synthetic Data Generation (GAN-tensorflow)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='nerdstudio',
    author_email='ashkan@nematifamilyfundation.onmicrosoft.com',
    url='https://github.com/osllmai/IndoxGen/tree/master/libs/indoxGen_tensor',
    keywords=[
        'AI',
        'deep learning',
        'language models',
        'synthetic data generation',
        'machine learning',
        'NLP'
    ],
    install_requires=[
        'dython==0.7.8',
        'libs==0.0.10',
        'scipy==1.14.1',
        'tensorflow==2.17.0',
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
