__author__ = 'euclides'
from setuptools import setup, find_packages
import os

setup(
    name="ml_utils",
    author="Euclides Fernandes Filho",
    author_email="euclides5414@gmail.com",
    package_dir={'': '../ml_utils'},
    packages=find_packages('../ml_utils'),
    version=open(os.path.join('ml_utils', 'resources', 'VERSION')).read(),
    install_requires=["scikit-learn", "pandas", "numpy", "xgboost"],
    dependency_links=[],
    include_package_data=True,
    zip_safe=False,
    test_suite="ml_utils.test"
)
