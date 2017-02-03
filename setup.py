from setuptools import setup, find_packages



setup(
    name='circusort',
    version='0.0.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'spyking-circus-ort = circusort:main',
        ]
    }
)
