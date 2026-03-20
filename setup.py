from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name             = 'geomagpro',
    version          = '1.0.0',
    author           = 'Muhammet Ali Aygün',
    author_email     = 'maygun@ogr.iu.edu.tr',
    description      = 'PSO, ABIC and Li-Oldenburg gravity and magnetic inversion suite',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url              = 'https://github.com/edmadz/Inversion-Module',
    packages         = find_packages(),
    python_requires  = '>=3.9',
    install_requires = [
        'numpy>=1.24',
        'scipy>=1.11',
        'matplotlib>=3.7',
        'pandas>=2.0',
    ],
    extras_require   = {
        'dev': ['pytest>=7.0', 'pytest-cov'],
    },
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords = ('#geophysics #gravity #magnetic #inversion #PSO #ABIC '
                '#Li-Oldenburg #Marmara Sea tectonic'),
)
