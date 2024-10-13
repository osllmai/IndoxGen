from setuptools import setup, find_packages
import os

# Get the absolute path of the current file
here = os.path.abspath(os.path.dirname(__file__))

# Print the path to check if it's correctly detecting the directory
print(f"Current directory: {here}")

# Read the requirements file
with open(os.path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

# Check if README.md exists and print a warning if not
readme_path = os.path.join(here, "README.md")
if not os.path.exists(readme_path):
    print(f"WARNING: README.md not found at {readme_path}")
else:
    # Read the README file for the long description
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name='indoxGen-tensor',
    version='0.0.8',
    license='AGPL-3.0-or-later',
    packages=find_packages(include=['indoxGen_tensor', 'indoxGen_tensor.*']),
    include_package_data=True,  # Includes additional files as per MANIFEST.in
    description='Indox Synthetic Data Generation (GAN-tensorflow)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='nerdstudio',
    author_email='ashkan@nematifamilyfundation.onmicrosoft.com',
    url='https://github.com/osllmai/IndoxGen/tree/master/libs/indoxGen_tensor',
    keywords=[
        'AI', 'deep learning', 'language models', 'synthetic data generation',
        'machine learning', 'NLP', 'GAN', 'tensorflow'
    ],
    install_requires=requirements,  # Dependencies from requirements.txt
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
