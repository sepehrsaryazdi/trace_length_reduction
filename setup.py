from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='trace_length_reduction',
   version='1.0',
   description='Trace Length Reduction Package',
   license="MIT",
   long_description=long_description,
   author='Sepehr Saryazdi',
   author_email='sepehr.saryazdi@sydney.edu.au',
   url="https://github.com/sepehrsaryazdi/trace_length_reduction",
   packages=find_packages('src',exclude=['tests']),  #same as name
   package_dir = {"": "src"},
   install_requires=['wheel', 'numpy', 'sympy', 'matplotlib', 'pandas'], #external packages as dependencies
   scripts=['main.py']
)