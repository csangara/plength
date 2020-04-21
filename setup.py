from setuptools import setup
from setuptools import find_packages

long_description = '''
Plength is a standalone file that helps measure the leaf, internode, and total length of rice seedlings.
'''

setup(name='Plength',
      version='0.14.2',
      description='Tool for measuring rice seedlings and coleoptiles',
      long_description=long_description,
      author='Chananchida Sang-aram',
      author_email='chananchida.sangaram@ugent.be',
      url='https://github.com/csangara/plength',
      install_requires=['numpy==1.14.1',
            'scipy==1.0.0',
			'scikit-image==0.13.1',
			'opencv-python==3.4.2.16',
			'matplotlib==3.0.0',
			'networkx==1.11',
			'numba==0.37.0',
			'Pillow==6.2.0'],
      dependency_links=['http://github.com/csangara/sknw/tarball/6594d8a953ecd8d4f961821b843b47a40391bb1b#egg=sknw'],
packages=find_packages())