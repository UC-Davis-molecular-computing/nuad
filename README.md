# dsd

## Introduction

dsd stands for "DNA sequence designer". It is a Python library that enables one to specify constraints on a DNA nanostructure made from synthetic DNA. It is not a standalone program, unlike other DNA sequence designers such as [NUPACK](http://www.nupack.org/design/new). Instead, it attempts to be more expressive than existing DNA sequence designers, at the cost of being less simple to use. The dsd library helps you to write your own DNA sequence designer, in case existing designers cannot capture the particular constraints of your project.

## API documentation
The API documentation is on readthedocs: https://dnadsd.readthedocs.io/


## Installation
dsd requires Python version 3.6 or higher. Currently, it cannot be installed using pip (see [issue #12](https://github.com/UC-Davis-molecular-computing/dsd/issues/12)). 

dsd uses [NUPACK](http://www.nupack.org/downloads) and [ViennaRNA](https://www.tbi.univie.ac.at/RNA/#download). While it is technically possible to use dsd without them, most of the pre-packaged constraints require them. ViennaRNA is fairly straightforward to install on any system.

You can download either NUPACK 3 or NUPACK 4. Installing NUPACK 3 requires compiling C code from source, and it is not straightforward to install on Windows (for example, using Cygwin or MinGW). To use NUPACK on Windows, you should use [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10), which essentially installs a command-line-only Linux inside of your Windows system.
Installing NUPACK 4 allows access to functions such as `pfunc4` and related functions
and can be done by following the installation instructions
in the online [user guide](https://piercelab-caltech.github.io/nupack-docs/start/).

If you are using Windows, you can then run python code calling the dsd library from either WSL (which will appear to the Python virtual machine as though it is running on Linux), or from the Windows command prompt/PowerShell directly. If you do the latter, then the dsd library, when making a call to NUPACK or ViennaRNA, will assume they have been installed under WSL and call them via the Windows command line, e.g., `wsl.exe -e RNAduplex ...`. This is what will happen, for instance, if you use an IDE such as [PyCharm](https://www.jetbrains.com/pycharm/) to run your Python program in Windows.

To install dsd:

1. Download the git repo, by one of two methods:
    - type `git clone https://github.com/UC-Davis-molecular-computing/dsd.git` at the command line, or
    - on the page `https://github.com/UC-Davis-molecular-computing/dsd`, click on Code &rarr; Download Zip:

      ![](images/screenshot-download-zip.png)

      and then unzip somewhere on your file system.

2. Add the directory `dsd` that you just created to your `PYTHONPATH` environment variable. In Linux, Mac, or [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) this is done by adding this line to your `~/.bashrc` file, where `/path/to/dsd` represents the path to the `dsd` directory:

    ```
    export PYTHONPATH="${PYTHONPATH}:/path/to/dsd"
    ```

    In Windows (outside of Windows Subsystem for Linux), the `PYTHONPATH` environment variable can be adjusted by right-clicking "This PC" on the desktop &rarr; Properties &rarr; Advanced System Settings &rarr; Environment Variables.

3. Install the PYthon packages dependencies listed in the file [requirements.txt](https://github.com/UC-Davis-molecular-computing/dsd/blob/main/requirements.txt) by typing `pip install numpy ordered_set psutil scadnano` at the command line.

4. Install NUPACK (version 3 and/or 4) and ViennaRNA following their installation instructions ([NUPACK](https://piercelab-caltech.github.io/nupack-docs/start/#installation-requirements) and [ViennaRNA](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html)). (If you do not install one of them, you can still install dsd, but most of the useful functions specifying pre-packaged constraints will be unavailable to call.) If installing on Windows, you must first install [Windows Subsystem for Linux (wsl)](https://docs.microsoft.com/en-us/windows/wsl/install-win10), and then install NUPACK and ViennaRNA from within wsl. After installing each of NUPACK 3 and/or ViennaRNA, add their executable directories to your `PATH` environment variable. (Similarly to how the `PYTHONPATH` variable is adjusted above.) NUPACK 4 does not come with an executable, so there is is no executable directory you need to add to `path`.

    To test that NUPACK 3 is installed correctly, type `pfunc` at the command line (the wsl command line if using Windows); the pfunc NUPACK 3 executable should be called.
    To test that NUPACK 4 is installed correctly, run `python3 -m pip show nupack`.
    To test that ViennaRNA is installed correctly, type `RNAduplex` at the command line.

    On Windows, you should also test that they can be called through the normal Windows command line (or Powershell) by typing `wsl -e pfunc` and `wsl -e RNAduplex`, since this is how the dsd library will call them if you run your Python programs from a Windows command line.

5. Test it works by typing `python` at the command line, then typing `import dsd`. It should import without errors:

    ```python
    $ python
    Python 3.7.5 (default, Nov  7 2019, 10:50:52)
    [GCC 8.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import dsd
    >>>
    ```

    To test that NUPACK and ViennaRNA can each be called from within the Python library (note that if you do not install both versions of NUPACK, and ViennaRNA, then only a subset of the following will succeed):

    ```python
    >>> import dsd.vienna_nupack as dv
    >>> dv.pfunc('GCGCGCGCGC') # test NUPACK 3
    -2.06056146
    >>> dv.pfunc4('GCGCGCGCGC') # test NUPACK 4
    -1.9079766874655928
    >>> dv.rna_duplex_multiple([('GCGCGCGCGC', 'GCGCGCGCGC')]) # test ViennaRNA
    [-15.7]
    ```

## Data model
dsd allows one to go from a design with abstract "domains", such as `a`, `a*`, `b`, `b*`, to concrete DNA sequences, for example, 
`a` = 5'-CCCAA-3', `a*` = 3'-GGGTT-5', `b` = 5'-AAAAAAC-3', `b*` = 3'-TTTTTTG-5', obeying the constraints you specify.

There are some pre-built constraints, for example limiting the number of G's in a domain or checking the partition function energy of a multi-domain strand (i.e., the strand's quantitative "secondary structure") according to the 3rd-party tools NUPACK and ViennaRNA. The user can also specify custom constraints.

In more detail, there are five main types of objects you create to describe your system:

- `Domain`: A `Domain` represents a contiguous subsequence of DNA. A single `Domain` represents both the DNA sequence and its complement. For instance there is one `Domain` with name `a`, with two versions: unstarred (`a`) and starred (`a*`). If the DNA sequence of `a` is 5'-CCCAA-3', then the DNA sequence of `a*` is 3'-GGGTT-5'. 

- `DomainPool`: Each `Domain` is assigned a single `DomainPool`, which can be thought of as a "source of DNA sequences" for the `Domain`. The designer will take DNA sequences from this source when attempting to find DNA sequences to assign to `Domain`'s to satisfy all constraints. Each `DomainPool` has a fixed length. Since each `Domain` only has one `DomainPool`, this means that each `Domain` has a fixed length as well. If no other constraints are specified, then the `DomainPool` simply provides all DNA sequences of that length. Though you will generally not call this method yourself, the method `DomainPool.generate_sequence()` returns a sequence from the pool. This method can be called infinitely many times (i.e., sequences can repeat, though the exact period after which they repeat is an unspecified implementation detail.)

- `Strand`: A `Strand` contains an ordered list `domains` of `Domain`'s, together with an identification of which `Domain`'s are starred in this `Strand`, the latter specified as a set `starred_domain_indices` of indices (starting at 0) into the list `domains`. For example, the `Strand` consisting of `Domain`'s `a`, `b*`, `c`, `b`, `d*`, in that order, would have `domains = [a, b, c, b, d]` and `starred_domain_indices = {1, 4}`.

- `Design`: This describes the whole system. Generally you will have one `Design`, which is composed of a list of `Strand`'s and lists of various types of `Constraint`'s, described below.

- `Constraint`: There are several kinds of constraint objects. Not all of them are related in the type hierarchy. 

    - **Sequence-level constraints:** 
    These are the strictest constraints, which do not even allow certain sequences to be considered. They are applied by a `DomainPool` before allowing a sequence to be returned from `DomainPool.generate_sequence()`. These are of two types: `NumpyConstraint` and `SequenceConstraint`. Each of them indicates whether a DNA sequence is allowed or not; for instance a constraint forbidding 4 G's in a row would permit AGGGT but forbid AGGGG. The difference between them is that a `NumpyConstraint` operates on many DNA sequences at a time, representing them as a 2D numpy byte array (e.g., a 100,000 &times; 15 array of bytes to represent 100,000 sequences, each of length 15), and for operations that numpy is suited for, can evaluate these constraints *much* faster than the equivalent Python code.

    - **Other constraints:**  All other constraints are subclasses of the abstract superclass `Constraint`. These constrains are "looser": sequences violating the constraints are allowed to be assigned to `Domain`'s. The sequence design algorithm steadily improves the design by changing sequences until all of these constraints are satisfied. The different subtypes correspond to different parts of the `Design` that are being evaluated by the `Constraint`. The types are:
        
        - `DomainConstraint`: This only looks at a single `Domain`. In practice this is not used much, since there's not much information in a `Domain` other than its DNA sequence, so a `SequenceConstraint` or `NumpyConstraint` typically would already have filtered out any DNA sequence not satisfying such a constraint.

        - `StrandConstraint`: This evaluates at a whole `Strand`. A common example is that NUPACK's `pfunc` should indicate a partition function energy above a certain threshold (indicating the `Strand` has little secondary structure.)

        - `DomainPairConstraint`: This evaluates a pair of `Domain`'s.

        - `StrandPairConstraint`: This evaluates a pair of `Strand`'s.

        - TODO: `ComplexConstraint` with a tuple of `Strand`'s?

        - `DomainsConstraint`, `StrandsConstraint`, `DomainPairsConstraint`, `StrandPairsConstraint`: These are similar to their singular counterparts listed above. The difference is that some checks may be faster to do in batch or parallel than one at a time. For instance, RNAduplex, an executable included with ViennaRNA, can examine many pairs of sequences, and it is much faster to give it all pairs at once, than to repeatedly call RNAduplex, once for each pair. 

        - `DesignConstraint`: This is rarely used in practice, but it can be used to express any constraint not captured by one of the constraints already listed. It takes the entire design as input.

    The "singular" constraints `DomainsConstraint`, `StrandsConstraint`, `DomainPairsConstraint`, `StrandPairsConstraint` each are given a function `evaluate`, which takes as input the relevant part of the design (e.g., a `StrandPairConstraint` takes as input two `Strand`'s) and returns a floating-point value. The interpretation of the returned value is as follows: if the constraint is satisfied, it should return 0.0. If the constraint is violated, it returns a positive number indicating "how much" the constraint is violated. The pre-packaged constraints are all of the form "*compare some energy value returned by NUPACK or ViennaRNA to a fixed threshold, and return the difference*". For example, with a threshold of -1.6 kcal/mol, if a `Strand` has partition function energy of -2.9 kcal/mol according to NUPACK's pfunc, then the constraint will return the value 1.3 = -1.6 - (-2.9), i.e., the actual energy is 1.3 beyond the threshold. If the actual energy is -1.2 instead, then it will return 0.0, since it is on the "good" side of the threshold. (The dsd sequence design algorithm actually converts all negative values to 0.0, so one could slightly simplify and simply return the value `threshold - energy`.)

    The "plural" constraints take as input a list of "`Design` parts" (e.g., a list of pairs of `Strand`'s for a `StrandPairsConstraint`). However, rather than returning a list of floats, the return value is slightly more general. Each of these constraints "blames" individual `Domain`'s each each violation. They TODO finish this.

