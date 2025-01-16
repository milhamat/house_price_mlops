from setuptools import setup, find_packages

setup(
    name="house_price_mlops",
    version="0.1",
    packages=find_packages(),  # Automatically finds all packages under `src`
    package_dir={"": "src"},  # Treat `src` as the root for packages
)
