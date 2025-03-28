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
   package_dir = {
            'trace_length_reduction': 'trace_length_reduction'},
   packages=['trace_length_reduction'],
   install_requires=['wheel', 'numpy', 'sympy', 'matplotlib', 'pandas', 'tk'], #external packages as dependencies
   scripts=['main.py']
)