import pathlib
import fnmatch

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as build_py_orig

package_root = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (package_root / 'README.md').read_text(encoding='utf-8')

class build_py_exclude_tests(build_py_orig):
    test_pattern = '*_test.py'
    def find_package_modules(self, *args, **kwargs):
        modules = super().find_package_modules(*args, **kwargs)
        return [
            (pkg, mod, filename)
            for (pkg, mod, filename) in modules
            if not fnmatch.fnmatchcase(filename, pat=self.test_pattern)
        ]

setup(
    name='correction',
    version='0.0.1',
    # TODO(mcsilla) 
    description='A sample Python project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # TODO(mcsilla) 
    # url='https://github.com/...',
    author='Csilla Majoros',
    author_email='',
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
    ],
    cmdclass={'build_py': build_py_exclude_tests},
    packages=find_packages(where=package_root),
    python_requires='>=3.5, <4',
    install_requires=[
        'numpy<1.19.0,>=1.18.5',
        'tensorflow==2.2.1',
        'tf-models-official==2.2.2',
        'tokenizers>=0.9.3',
        'transformers>=3.0.2',
    ],

    extras_require={},
    package_data={},
    # data_files=[],
    entry_points={
        'console_scripts': [
            'generate_correction_dataset=correction.correction_dataset_generator_main:main',
        ],
    },
    project_urls={
        # TODO(mcsilla) 
        # 'Source': 'url='https://github.com/...',
    },
)