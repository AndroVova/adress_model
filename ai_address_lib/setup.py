from setuptools import setup, find_packages

setup(
    name='ai_address_lib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.16.1',       
        'transformers>=4.44.2',     
        'numpy>=1.26.4',            
        'h5py>=3.11.0',           
        'scipy>=1.14.1',
        'tf-keras', 
        'sentencepiece',
        'scikit-learn',
        'tqdm',
        'pandas',
    ],
    description='Library for adress model predictions',
    author='vova',
    author_email='vova.safoschin@gmail.com'
)
