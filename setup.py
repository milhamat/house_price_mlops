from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> List[str]:
    """
    Function will return the list of requirements
    """
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="house_price_mlops",
    version="0.1",
    packages=find_packages(),  # Automatically finds all packages under `src`
    # package_dir={"": "src"},  # Treat `src` as the root for packages
    # install_requires =get_requirements('requirements.txt')
)
