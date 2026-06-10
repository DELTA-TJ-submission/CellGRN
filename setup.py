from setuptools import setup, find_packages


author = (
    "Shaliu Fu"
)

setup(
    author=author,
    author_email="adam.tongji@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: bioinformatics",
    ],
    description="CellGRN ",
    license="MIT license",
    # long_description=readme + "\n\n" + history,
    # include_package_data=True,
    keywords="cellgrn",
    name="cellgrn",
    packages=find_packages(),
    install_requires=[
        "anndata",
        "numpy",
        "pandas",
        "pyranges",
        "scanpy",
        "scikit-learn",
        "scipy",
        "umap-learn",
    ],
    version="0.0.0",
    zip_safe=False,
)
