from setuptools import setup

setup(
    name='geox',
    version='0.1.0',    
    description='geox package',
    url='https://gitlab.lisn.upsaclay.fr/salimath/geox',
    author='Shwetha Salimath',
    author_email='shwetha.salimath@lisn.fr',
    license='CC BY-NC-SA',
    packages=['geox'],
    install_requires=['pandas',
                      'numpy', 
                      'fastparquet',
                      'tsai',
                      'fastai',
                      'statistics',
                      'pathlib',
                      'matplotlib',
                      'torch',
                      'scikit-learn',
                      'dtaidistance',
                      'threadpoolctl'                    
                      ],

    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: CC BY-NC-SA', 
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)