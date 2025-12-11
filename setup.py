# setup.py
from setuptools import setup, find_packages

setup(
    name="puzzlegame",
    version="0.1.0",
    package_dir={"": "src"},  # 指定包目录在 src 下
    packages=find_packages(where="src"),  # 在 src 中查找包
)