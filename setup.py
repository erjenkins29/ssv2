#!/user/bin/env python
# -*- coding: utf-8 -*-

import sys
from setuptools import setup, find_packages

setup(
        author='Evan',
        author_email='evan@earlydata.com',
        name='ss_project',
        version='1.0',
        packages=find_packages(),
        package_data={'':['*.txt','*.jar'],}
)


