# !/usr/bin/env python
# -*- coding: utf-8 -*-


def main():
    from setuptools import setup, find_packages

    version_dict = {}
    init_filename = "squirm/version.py"
    exec(
        compile(open(init_filename, "r").read(), init_filename, "exec"),
        version_dict)

    setup(name="squirm",
          version=version_dict["VERSION_TEXT"],
          description=("Python abstraction layer for cluster job schedulers"),
          long_description=open("README.md", "rt").read(),
          author="CEESD",
          author_email="mjsmith6@illinois.edu",
          license="MIT",
          url="https://github.com/majosm/squirm",
          classifiers=[
              "Development Status :: 1 - Planning",
              "Intended Audience :: Developers",
              "Intended Audience :: Other Audience",
              "Intended Audience :: Science/Research",
              "License :: OSI Approved :: MIT License",
              "Natural Language :: English",
              "Programming Language :: Python",
              "Programming Language :: Python :: 3.6",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.8",
              "Programming Language :: Python :: 3.9",
              "Topic :: Scientific/Engineering",
              "Topic :: Software Development :: Libraries",
              "Topic :: Utilities",
              ],

          packages=find_packages(),

          python_requires="~=3.6",

          install_requires=[
              "mpi4py>=3",
              "pytest>=2.3"
              ])


if __name__ == "__main__":
    main()
