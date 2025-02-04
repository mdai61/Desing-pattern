from setuptools import setup, find_packages

setup(
    name="project_name",  # Replace with your project name
    version="0.1.0",
    author="Your Name",
    author_email="mdai61@example.com",
    description="A machine learning project with Kivy UI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mdai61/eDo-Desing-pattern",  # Replace with actual repo link
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "xgboost",
        "matplotlib",
        "scikit-learn",
        "kivy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "project_name=main:main",  # Allows running from terminal using "eDo-Desing-pattern"
        ],
    },
)
