import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="top_pr",
    version="0.1",
    author="Pumjun Kim",
    description="TopP&R: Robust Support Estimation Approach "
                "for Evaluating Fidelity and Diversity in Generative Models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LAIT-CVLab/TopPR",
    packages=setuptools.find_packages(),
    keywords=['toppr', 'evaluation metric', 'topological metric', 'precision and recall'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'scikit-learn',
        'scipy',
        'tqdm',
        'matplotlib',
        'gudhi'
    ],
)