#!/usr/bin/env python3

import os



dir = input("Enter the project name: ")
pkg_name = input("Enter a name for the pkg: ")

os.system("mkdir " + dir)
os.chdir(dir)
# create src and its components
os.system(f"mkdir src/ && cd src/ && mkdir {pkg_name} && cd {pkg_name} && touch __init__.py && touch {dir}.py")
# create need files
os.system("touch README.md && mkdir tests/")
build = """
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mlib"
version = "0.0.1"
authors = [
  { name="Mohamed Toumi", email="hamatm662@gmail.com" },
]
description = "A from scratch ML package"
readme = "README.md"
requires-python = ">=3.7"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[project.urls]
"Homepage" = "https://github.com/ZeroDay0utplay/Machine-Learning-Library"
"Bug Tracker" = "https://github.com/ZeroDay0utplay/Machine-Learning-Library/issues"
"""
with open("pyproject.toml", "w") as b:
    b.write(build)


license = """
Copyright (c) 2023 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

with open("LICENSE", "w") as l:
    l.write(license)


