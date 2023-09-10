import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

setuptools.setup(
    name="AutoGrid",
    version="0.0.1",
    author="Pablo Torres Anaya",
    author_email="pablo.ta@ugr.com",
    description="Experiment automatization for Grid2Op",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ugr-sail/AutoGrid",
    project_urls={
        "Bug Tracker": "https://github.com/ugr-sail/AutoGrid/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_reqs=requirements
)
