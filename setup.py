from setuptools import setup, find_packages

setup(
    name="my_data_viz_package",
    version="0.1.0",
    description="A simple data visualization package for teaching purposes.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Mohamed Warsame",
    author_email="your.email@example.com",
    url="https://github.com/mohwarsame273/data_viz_package",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas",
        "upsetplot",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
