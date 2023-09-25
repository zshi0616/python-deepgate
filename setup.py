from setuptools import setup, find_packages
setup(
    name = 'deepgate', 
    version = '2.0.1',
    description= 'DeepGate: Learn Logic Gate Representation', 
    author = 'Zhengyuan (Stone) Shi', 
    author_email= 'zyshi21@cse.cuhk.edu.hk', 
    packages = find_packages(exclude=['examples', 'data']),
    include_package_data = True,
)