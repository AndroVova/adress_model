from setuptools import setup, find_packages

setup(
    name='ai_address_lib',
    version='0.4.1',
    description='Library for address model predictions',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Vova Safoshyn',
    author_email='vova.safoschin@gmail.com',
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
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
