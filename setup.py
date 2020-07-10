from setuptools import setup

setup(
    name="pynsga3",
    version="0.0.1",
    package_dir={"": "src"},
    packages=["pynsga3", "pynsga3.operators", "pynsga3.utils"],
    install_requires=[
        'numpy>=1.10.*',
        "scipy>=0.15.*"
    ],
    author="Alexander Kryuchkov",
    author_email="KernelA@users.noreply.github.com",
    description="Python implementation of NSAG-3 algorithm.",
    long_description="Deb, Kalyanmoy & Jain, Himanshu. (2014)."
                     " An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach,"
                     " Part I: Solving Problems With Box Constraints."
                     " Evolutionary Computation, IEEE Transactions on. 18. 577-601. 10.1109/TEVC.2013.2281535.",
    license="MIT",
    url="https://github.com/KernelA/nsga3",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha"
    ]
)

