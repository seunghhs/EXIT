from setuptools import setup, find_packages

__version__ = "1.0.0"

setup(
    name="EXIT",
    version="1.0.0",
    license="KAIST",
    author="Seunghee Han",
    author_email="sisifhro@kaist.ac.kr",
    description="EXIT: Experimental XRD Integrated Transformer for MOF property prediction",
    packages=find_packages(),
    package_data={"exit": ["tokenizer/vocab_new.txt", "config/*.yml"]},
    python_requires=">=3.8",
)
