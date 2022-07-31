# nuad

nuad is a Python library that enables one to specify constraints on a DNA (or RNA) nanostructure made from synthetic DNA/RNA and then attempts to find concrete DNA sequences that satisfy the constraints.

Note: If you are reading this on the PyPI website, many links below won't work. They are relative links intended to be read on the [GitHub README page](https://github.com/UC-Davis-molecular-computing/nuad/tree/main#readme).

## Table of contents

* [Overview](#overview)
* [API documentation](#api-documentation)
* [Installation](#installation)
  * [Installing nuad](#installing-nuad)
  * [Installing NUPACK and ViennaRNA](#installing-nupack-and-viennarna)
* [Data model](#data-model)
* [Constraint evaluations must be pure functions of their inputs](#constraint-evaluations-must-be-pure-functions-of-their-inputs)
* [Examples](#examples)
* [Parallelism](#parallelism)
* [Reporting issues](#reporting-issues)
* [Contributing](#contributing)
* [NUPACK Copyright Notice](#nupack-copyright-notice)

## Overview

nuad stands for "NUcleic Acid Designer".† It is a Python library that enables one to specify constraints on a DNA (or RNA) nanostructure made from synthetic DNA/RNA (for example, "*all strands should have complex free energy at least -2.0 kcal/mol according to [NUPACK](http://www.nupack.org/)*", or "*every binding domain should have binding energy with its perfect complement between -8.0 kcal/mol and -9.0 kcal/mol in the [nearest-neighbor energy model](https://en.wikipedia.org/wiki/Nucleic_acid_thermodynamics#Nearest-neighbor_method)*"), and then attempts to find concrete DNA sequences that satisfy the constraints. It is not a standalone program, unlike other DNA sequence designers such as [NUPACK](http://www.nupack.org/design/new). Instead, it attempts to be more expressive than existing DNA sequence designers, at the cost of being less simple to use. The nuad library helps you to write your own DNA sequence designer, in case existing designers cannot capture the particular constraints of your project.

Note: The nuad package was originally called dsd (DNA sequence designer), so you may see some old references to this name for the package.

†A secondary reason for the name of the package is that some work was done when the primary author was on sabbatical in Maynooth, Ireland, whose original Irish name is [*Maigh Nuad*](https://en.wikipedia.org/wiki/Maynooth#Etymology).


## API documentation
The API documentation is on readthedocs: https://nuad.readthedocs.io/


## Installation
nuad requires Python version 3.7 or higher. Currently, although it can be installed using pip by typing `pip install nuad`, it depends on two pieces of software that are not installed automatically by pip (see [issue #12](https://github.com/UC-Davis-molecular-computing/nuad/issues/12)). 

nuad uses [NUPACK](http://www.nupack.org/downloads) and [ViennaRNA](https://www.tbi.univie.ac.at/RNA/#download), which must be installed separately (see below for link to installation instructions). While it is technically possible to use nuad without them, most of the pre-packaged constraints require them.

To use NUPACK on Windows, you must use [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install), which essentially installs a command-line-only Linux inside of your Windows system, which has access to your Windows file system. If you are using Windows, you can then run python code calling the nuad library from WSL (which will appear to the Python virtual machine as though it is running on Linux). WSL is necessary to use any of the constraints that use NUPACK 4.

### Installing nuad

To install nuad, you can either install it using pip (the slightly simpler option) or git. No matter which method you choose, you must also install NUPACK and ViennaRNA separately (see [instructions below](#installing-nupack-and-viennarna)).

- pip
  
  At the command line (WSL for Windows, not the Powershell prompt), type

  ```
  pip install nuad
  ```

- git

  This method has more steps, but it might be preferable if you want to use a new feature that is not on the main branch: the package installed by the pip instructions above will install the version currently on the main branch.

  1. Download the git repo, by one of two methods:
      - Install [git](https://git-scm.com/downloads) if necessary, then type 
    
          ```git clone https://github.com/UC-Davis-molecular-computing/nuad.git``` 
    
        at the command line, or
      - on the page `https://github.com/UC-Davis-molecular-computing/nuad`, click on Code &rarr; Download Zip:

        ![](images/screenshot-download-zip.png)

        and then unzip somewhere on your file system.

  2. Add the directory `nuad` that you just created to your `PYTHONPATH` environment variable. In Linux, Mac, or [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install-win10), this is done by adding this line to your startup script (e.g., `~/.bashrc`, or `~/.bash_profile` for Mac OS), where `/path/to/nuad` represents the path to the `nuad` directory:

      ```
      export PYTHONPATH="${PYTHONPATH}:/path/to/nuad"
      ```


  3. Install the Python packages dependencies listed in the file [requirements.txt](https://github.com/UC-Davis-molecular-computing/nuad/blob/main/requirements.txt) by typing 

      ```
      pip install numpy ordered_set psutil pathos xlwt xlrd tabulate scadnano
      ``` 
    
      at the command line. If you have Python 3.7 then you will also have to install the `typing_extensions` package: `pip install typing_extensions`

### Installing NUPACK and ViennaRNA

Recall that if you are using Windows, you must do all installation through [WSL](https://docs.microsoft.com/en-us/windows/wsl/install) (Windows subsystem for Linux).

Install NUPACK (version 4) and ViennaRNA following their installation instructions ([NUPACK installation](https://docs.nupack.org/start/#maclinux-installation), [ViennaRNA installation](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html), and [ViennaRNA downloads](https://www.tbi.univie.ac.at/RNA/#download)). If you do not install one of them, you can still install nuad, but most of the useful functions specifying pre-packaged constraints will be unavailable to call.

After installing ViennaRNA, it may be necessary to add its executables directory (the directory containing executable programs such as RNAduplex) to your `PATH` environment variable. (Similarly to how the `PYTHONPATH` variable is adjusted above.) NUPACK 4 does not come with an executable, so this step is unnecessary; it is called directly from within Python.

<!-- To test that NUPACK 4 is installed correctly, run `python3 -m pip show nupack`. -->

<!-- To test that ViennaRNA is installed correctly, type `RNAduplex` at the command line. -->

Type `python` at the command line, then type `import nuad`. It should import without errors:

```python
$ python
Python 3.7.5 (default, Nov  7 2019, 10:50:52)
[GCC 8.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import nuad
>>>
```

To test that NUPACK and ViennaRNA can each be called from within the Python library (note that if you do not install NUPACK and/or ViennaRNA, then this will fail):

```python
>>> import nuad.vienna_nupack as nv
>>> nv.pfunc('GCGCGCGCGC')  # test NUPACK 4
-1.9079766874655928
>>> nv.rna_duplex_multiple([('GCGCGCGCGC', 'GCGCGCGCGC')]) # test ViennaRNA
[-15.7]
```

## Data model
nuad allows one to go from a design with abstract "domains", such as `a`, `a*`, `b`, `b*`, to concrete DNA sequences, for example, 
`a` = 5'-CCCAA-3', `a*` = 3'-GGGTT-5', `b` = 5'-AAAAAAC-3', `b*` = 3'-TTTTTTG-5', obeying the constraints you specify.

There are some pre-built constraints, for example limiting the number of G's in a domain or checking the complex free energy of a strand (i.e., the strand's quantitative amount of "secondary structure") according to the 3rd-party tools [NUPACK](http://www.nupack.org) and [ViennaRNA](https://www.tbi.univie.ac.at/RNA). The user can also specify custom constraints.

In more detail, there are five main types of objects you create to describe your system:

- `Domain`: A `Domain` represents a contiguous subsequence of DNA. A single `Domain` represents both the DNA sequence and its complement. For instance there is one `Domain` with name `a`, with two versions: unstarred (`a`) and starred (`a*`). If the DNA sequence of `a` is 5'-CCCAA-3', then the DNA sequence of `a*` is 3'-GGGTT-5'. 

- `DomainPool`: Each `Domain` is assigned a single `DomainPool`, which can be thought of as a "source of DNA sequences" for the `Domain`. (See exceptions below.) The sequence design algorithm will take DNA sequences from this source when attempting to find DNA sequences to assign to `Domain`'s to satisfy all constraints. Each `DomainPool` has a fixed length. Since each `Domain` only has one `DomainPool`, this means that each `Domain` has a fixed length as well. If no other constraints are specified, then the `DomainPool` simply provides all DNA sequences of that length. Though you will generally not call this method yourself, the method `DomainPool.generate_sequence()` returns a sequence from the pool. This method can be called infinitely many times (i.e., sequences can repeat, though the exact period after which they repeat is an unspecified implementation detail.)

  - There are two types of `Domain`'s with no associated `DomainPool`. One type is a `Domain` with the field `fixed` set to `True` by calling the method `Domain.set_fixed_sequence()`, which has some fixed DNA sequence that cannot be changed. A fixed `Domain` has no `DomainPool`.)

  - The other type is a `Domain` with the field `dependent` set the `True` (by assigning the field directly). Such a domain is dependent for its sequence on the sequence of some other `Domain` with `dependent = False` that either contains it as a subsequence, or is contained in it as a subsequence. For example, one can declare the domain `a` is independent (has `dependent = False`), with length 8, and has dependent subdomains `b` and `c` of length 5 and 3. `a` would have a `DomainPool`, and if `a` is assigned sequence AAACCGTT, then `b` is automatically assigned sequence AAACC, and `c` is automatically assigned sequence GTT. Such subdomains are assigned via the field `Domain.subdomains`; see the API documentation for more details: https://dnadsd.readthedocs.io/en/latest/#constraints.Domain.dependent and https://dnadsd.readthedocs.io/en/latest/#constraints.Domain.subdomains.

- `Strand`: A `Strand` contains an ordered list `domains` of `Domain`'s, together with an identification of which `Domain`'s are starred in this `Strand`, the latter specified as a set `starred_domain_indices` of indices (starting at 0) into the list `domains`. For example, the `Strand` consisting of `Domain`'s `a`, `b*`, `c`, `b`, `d*`, in that order, would have `domains = [a, b, c, b, d]` and `starred_domain_indices = {1, 4}`.

- `Design`: This describes the whole system. Generally you will have one `Design`, which is composed of a list of `Strand`'s.

- `Constraint`: There are several kinds of constraint objects. Not all of them are related in the type hierarchy. 

    - **"hard" constraints on Domain sequences:** 
    These are the strictest constraints, which do not even allow certain `Domain` sequences to be considered. They are applied by a `DomainPool` before allowing a sequence to be returned from `DomainPool.generate_sequence()`. These are of two types: `NumpyConstraint` and `SequenceConstraint`. Each of them indicates whether a DNA sequence is allowed or not; for instance a constraint forbidding 4 G's in a row would permit AGGGTT but forbid AGGGGT. The difference between them is that a `NumpyConstraint` operates on many DNA sequences at a time, representing them as a 2D numpy byte array (e.g., a 1000 &times; 15 array of bytes to represent 1000 sequences, each of length 15), and for operations that numpy is suited for, can evaluate these constraints *much* faster than the equivalent Python code that would loop over each sequence individually. However, if you have a constraint that is not straightforward to express using numpy operations, then a `SequenceConstraint` can be used to express it in plain Python. A `SequenceConstraint` is simply a type alias for a Python function that takes a string as input representing the DNA sequence and returns a Boolean indicating whether the sequence satisfies the constraint. Due to the speed of numpy, it is advised to use `SequenceConstraint`'s only if necessary because it cannot be expressed as a `NumpyConstraint`.

    - **"soft" constraints:**  All other constraints are subclasses of the abstract superclass `Constraint`. These constrains are "softer": sequences violating the constraints are allowed to be assigned to `Domain`'s. The sequence design algorithm steadily improves the design by changing sequences until all of these constraints are satisfied. The different subtypes of the base class `Constraint` correspond to different parts of the `Design` that are being evaluated by the `Constraint`. The types are:

        - `SingularConstraint`: This is an abstract superclass of the following concrete subclasses. The difference with the other abstract superclass `BulkConstraint` is explained in `BulkConstraint` below.
        
            - `DomainConstraint`: This only looks at a single `Domain`. In practice this is not used much, since there's not much information in a `Domain` other than its DNA sequence, so a `SequenceConstraint` or `NumpyConstraint` typically would already have filtered out any DNA sequence not satisfying such a constraint.

            - `StrandConstraint`: This evaluates a whole `Strand`. A common example is that NUPACK's `pfunc` should indicate a complex free energy above a certain threshold, indicating the `Strand` has little secondary structure. This example constraint is available in the library by calling [nupack_strand_complex_free_energy_constraint](https://dnadsd.readthedocs.io/en/latest/#constraints.nupack_strand_complex_free_energy_constraint).

            - `DomainPairConstraint`: This evaluates a pair of `Domain`'s.

            - `StrandPairConstraint`: This evaluates a pair of `Strand`'s.

            - `ComplexConstraint`: This evaluates a tuple of `Strand`'s of arbitrary size.

        - `BulkConstraint`: The subclasses of `SingularConstraint` discussed above each evaluate a single part of the design at a time. The classes `DomainsConstraint`, `StrandsConstraint`, `DomainPairsConstraint`, `StrandPairsConstraint`, `ComplexesConstraint` are subclasses of `BulkConstraint`. The difference is that some checks may be faster to do in batch or parallel than one at a time. For instance, RNAduplex, an executable included with ViennaRNA, can examine many pairs of sequences, and it is much faster to give it all pairs at once in a single call to RNAduplex, than to repeatedly call RNAduplex from a Python loop, once for each pair. 

        - `DesignConstraint`: This is rarely used in practice, but it can be used to express any constraint not captured by one of the constraints already listed. It takes the entire design as input.

    The `SingularConstraint` subclasses `DomainsConstraint`, `StrandsConstraint`, `DomainPairsConstraint`, `StrandPairsConstraint`, and `ComplexConstraint` each are given a function `evaluate`, which takes as input the relevant part of the design (e.g., a `StrandPairConstraint` takes as input two `Strand`'s; technically the input is a bit more complex; see [Parallelism](#parallelism) below for details.). `evaluate` returns a pair `(excess, summary)`, where `excess` is floating-point value and `summary` is a string.
    
    The interpretation of `excess` is as follows: if the constraint is satisfied, `excess` should be 0.0. If the constraint is violated, `excess` should be a positive number indicating "how much" the constraint is violated. The pre-packaged constraints are mostly of the form "*compare some numeric value returned by NUPACK or ViennaRNA to a fixed threshold, and return the difference*". For example, with a threshold of -1.6 kcal/mol (i.e., we want all strands to have complex free energy greater than or equal to -1.6 kcal/mol), if a `Strand` has complex free energy of -2.9 kcal/mol according to NUPACK's pfunc, then the `excess` will be 1.3 = -1.6 - (-2.9), since the actual energy is 1.3 beyond the threshold. If the actual energy is -1.2 instead of -2.9, then it will return 0.0, since it is on the "good" side of the threshold. (The nuad sequence design algorithm actually converts all negative values to 0.0, so one could simply return the value `threshold - energy`.)

    The second returned value `summary` is a 1-line string briefly summarizing the result of evaluating the constraint. The pre-packaged constraint that evaluate's a `Strand`'s complex free energy with NUPACK, in the example above, returns `summary = "-2.9 kcal/mol"`. The `summary` is used in automatically generating reports on the constraints during the search, so that the user can inspect how well the search is doing.

    The `BulkConstraint` subclasses `DomainsConstraint`, `StrandsConstraint`, `DomainPairsConstraint`, `StrandPairsConstraint` use a different function, called `evaluate_bulk`, which take as input a list of "`Design` parts" (e.g., a list of pairs of `Strand`'s for a `StrandPairsConstraint`). The return value is of type `List[Tuple[DesignPart, float, str]]`, i.e., a list of triples, where each triple is `(part, excess, summary)`.
    
    `part` is the individual part of the design that caused a problem. Generally it will be one of the elements of the list passed to `evaluate_bulk`, though the returned list could be smaller than the input list. This is because some parts may satisfy the constraint, and generally the only parts returned from `evaluate_bulk` are those that violated the constraint. `excess` and `summary` have the same interpretation as with the "singular" constraints.

    The search algorithm evaluates the constraints, and for each violated constraint, it turns the `excess` value into a "score" by first passing it through the "score transfer function", which by default squares the value, and then multiplies by the value `Constraint.weight` (by default 1). The goal of the search is to minimize the sum of scores across all violated `Constraint`'s. The reason that the score is squared is that this leads the search algorithm to (slightly) favor reducing the excess of constraint violations that are "more in excess", i.e., it would reduce the total score more to reduce an excess from 4 to 3 (reducing the score from 4<sup>2</sup>=16 to 3<sup>2</sup>=9, a reduction of 16-9=7) than to reduce an excess from 2 to 1 (which reduces 2<sup>2</sup>=4 to 1<sup>2</sup>=1, a reduction of only 4-1=3).

    The full search algorithm is described in the [API documentation for the function nuad.search.search_for_dna_sequences](https://dnadsd.readthedocs.io/en/latest/#search.search_for_dna_sequences).


## Constraint evaluations must be pure functions of their inputs

For all constraints, it is critical that the `evaluate` or `evaluate_bulk` functions be *pure* functions of their inputs: the return value should depend only on the parameters passed to the function. For example, a `StrandPairConstraint` takes two strands as input, and its `(excess, summary)` return values should depend *only* on those two strands. Similarly, a `StrandsConstraint`, whose `evaluate_bulk` function takes a list of strands as input, should return a list of tuples, where each tuple represents a violation of a strand that depends only on that strand. This is required because nuad does an optimization in which constraints are only evaluated if they depend on parts of the design that contain the domain(s) that changed in the current iteration.

For example, suppose there are 100 strands, but only 3 strands contain the domain `x`, and `x` is the domain whose DNA sequence is changed in the current search iteration. Then each `StrandConstraint` `s` will be evaluated only on those 3 strands, on the assumption that the other 97 strands would have the same output of the function `s.evaluate` as before. 

In the case of `evaluate_bulk`, the constraint is even stronger. `evaluate_bulk` takes a list of objects as input and returns of list of the same size. The `i`'th element of the returned list should depend only on the `i`'th element of the input list. This is because a similar optimization is done as above. For example, if the changed domain appears in only 3 strands out of 100, a `StrandsConstraint` will pass in only those 3 strands as input to `evaluate_bulk`, not the full list of 100 strands, on the assumption that the other 97 strands would be processed the same by the `evaluate_bulk` function.

## Examples
Some example scripts can be found in the [examples/](examples/) subfolder.

In particular, the example [sst_canvas.py](examples/sst_canvas.py) shows a fairly simple design with realistic constraints for designing DNA sequences for a 2D canvas of single-stranded tiles (SSTs), similar to the sort of design from [this paper](https://doi.org/10.1038/nature11075).


## Parallelism
Each "singular" constraint in nuad includes the ability to specify the Boolean field `Constraint.parallel`. If True, then the various parts of the Design will be evaluated in parallel, taking advantage of multi-core systems. In practice, we have found that the overhead associated with doing this is fairly hefty, so it is unlikely that one would see, for example, an 8-fold speedup on an 8-core system. However, one could potentially see a speedup of 2x or 3x.

Given the nature of the stochastic local search, we have found that it is often a more effective and low-overhead use of multiple cores to simply start many independent instances of the designer, each with `Constraint.parallel` = False for all of the `Constraint`'s. One reason this approach is preferable is that, with very tight constraints that are difficult to satisfy, many runs of the designer will get stuck in local minima, and one often will simply pick the run that got stuck in the lowest minimum.

However, the parallelism feature is there if desired, and it is the reason that the `evaluate` function described in [the Data model section](#data-model) above take slightly more complicated arguments than hinted in that section, which we explain next.

Each "singular" constraint `DomainsConstraint`, `StrandsConstraint`, `DomainPairsConstraint`, `StrandPairsConstraint`, or `ComplexConstraint` is specified primarily by a function called `evaluate` that takes two major types of arguments: DNA sequence(s) of the related design part(s), and optional arguments containing the part(s) themselves. For example, a `StrandConstraint`'s `evaluate` function takes as input a tuple of strings of length 1, containing the DNA sequence corresponding to the `Strand` object, as well as the `Strand` object themselves. For concreteness, we stick with this example of `StrandConstraint`, but the idea applies to the other singular constraints above.
    
The type of the `Strand` parameter is actually `Optional[Strand]`, so could have the value `None`, for the following reason. If you use nuad's automatic parallelization feature (by setting a Constraint's `parallel` field to True), then when constraints such as this are called, the `Strand`'s will not be given to the `evaluate` function, only the DNA sequences. This is because the [pathos](https://pypi.org/project/pathos/) library is used for parallelization, which uses the [dill](https://pypi.org/project/dill/) library to serialize objects for parallel processing. In practice it takes much longer to serialize the entire `Strand` object than only its DNA sequence. The upshot is that if you don't use parallelization, then you can write constraints that reference not only the DNA sequence of a Strand or other part of the design, but the object representing the part itself. However, to use parallelization, only the DNA sequence of the part will be available to evaluate the constraint, and the second argument representing the part itself will be `None`.

The `evaluate_bulk` function for "plural" constraints `DomainsConstraint`, `StrandsConstraint`, `DomainPairsConstraint`, and `StrandPairsConstraint` does not have this issue, since there is no automatic parallelization feature for "plural" constraints. Therefore it simply takes as input a list of the design "parts" (e.g., list of `Strand`'s, list of pairs of `Domain`'s, etc.) to be evaluated.


## Reporting issues

Please report issues (bugs or feature requests) at the [nuad GitHub repository issues page](https://github.com/UC-Davis-molecular-computing/nuad/issues).


## Contributing

See the [CONTRIBUTING document](CONTRIBUTING.md).

## NUPACK Copyright Notice

Since nuad will eventually be distributed with NUPACK, we include the following license
agreement as required by [NUPACK](http://www.nupack.org/downloads/register).

### NUPACK Software License Agreement for Non-Commercial Academic Use and Redistribution
Copyright © 2022 California Institute of Technology. All rights reserved.

1. Use and redistribution in source form and/or binary form, with or without modification, are permitted for non-commercial academic purposes only, provided that the following conditions are met:

2. Redistributions in source form must retain the above copyright notice, this list of conditions and the following disclaimer.

3. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

4. Web applications that use the software in source form or binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in online documentation provided with the web application.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote derivative works without specific prior written permission.

### Disclaimer
This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall the copyright holder or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
