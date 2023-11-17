from setuptools import setup

DISTNAME = "voromesh"

VERSION = "0.1.0"

INSTALL_REQUIRES = ["numpy", "shapely>=2.0.0", "scipy", "pyvista", "vtk", "toughio", "h5py", "numba"]

PACKAGES = ["voromesh", "tests"]

METADATA = dict(
    name=DISTNAME,
    version=VERSION,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    )

def setup_package():
    setup(**METADATA)


if __name__ == "__main__":
    setup_package()
