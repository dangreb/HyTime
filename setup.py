import setuptools

setuptools.setup(
    name="HyTime",
    version="0.1",
    author="dasgreb",
    url="http://github.com/dasgreb",
    description="Hydea Neural Networks - Time Utilities",
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.10.13',
    install_requires=[],
    entry_points={
        'console_scripts': [
            'hytm = hytm.__main__:main',
        ]
    },
)