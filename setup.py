from setuptools import setup
from os import path


# this is ugly, but appears to be standard practice:
# https://stackoverflow.com/questions/17583443/what-is-the-correct-way-to-share-package-version-with-setup-py-and-the-package/17626524#17626524
def extract_version(filename: str):
    with open(filename) as f:
        lines = f.readlines()
    version_comment = '# version line; WARNING: do not remove or change this line or comment'
    for line in lines:
        if version_comment in line:
            idx = line.index(version_comment)
            line_prefix = line[:idx]
            parts = line_prefix.split('=')
            parts = [part.strip() for part in parts]
            version_str = parts[-1]
            version_str = version_str.replace('"', '')
            version_str = version_str.replace("'", '')
            version_str = version_str.strip()
            return version_str
    raise AssertionError(f'could not find version in {filename}')


version = extract_version('nuad/__version__.py')
print(f'nuad version = {version}')


with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='nuad',
      packages=['nuad'],
      version=version,
      license='MIT',
      description="nuad stands for \"NUcleic Acid Designer\". Enables one to specify constraints on a DNA (or RNA) nanostructure made from synthetic DNA/RNA and then attempts to find concrete DNA sequences that satisfy the constraints.",
      author="David Doty",
      author_email="doty@ucdavis.edu",
      url="https://github.com/UC-Davis-molecular-computing/nuad",
      long_description=long_description,
      long_description_content_type='text/markdown; variant=GFM',
      python_requires='>=3.7',
      install_requires=install_requires,
      include_package_data=True,
      )
