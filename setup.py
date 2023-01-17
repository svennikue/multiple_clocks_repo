from setuptools import find_packages, setup

setup(
    name="mc",
    description = "These are all scripts to run the multiple clocks project.",
    long_description= """In particular, there are several sub-modules that do different things. 
    Firstly, the simulation sub-module allows to create a 3x3 grid with 4 rewards on it, and 
    codes predictions of neural firing pattern for space and multiple phase clocks.""",
    packages=find_packages(),
)
