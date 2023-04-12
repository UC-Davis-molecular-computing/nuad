# Contributing to the nuad Python package 
First off, thanks for taking the time to contribute!

The following is a set of guidelines for contributing to nuad.
Feel free to propose changes to this document in a pull request, 
or post questions as issues on the [issues page](https://github.com/UC-Davis-molecular-computing/nuad/issues).



## Table of contents

* [What should I know before I get started?](#what-should-i-know-before-i-get-started)
* [Cloning](#cloning)
* [Pushing to the repository dev branch and documenting changes (done on all updates)](#pushing-to-the-repository-dev-branch-and-documenting-changes-done-on-all-updates)
* [Pushing to the repository main branch and documenting changes (done less frequently)](#pushing-to-the-repository-main-branch-and-documenting-changes-done-less-frequently)
* [Styleguide](#styleguide)
* [Miscellaneous](#miscellaneous)


## What should I know before I get started?

### Python
First, read the [README](README.md) to familiarize yourself with the package from a user's perspective.

The nuad Python package requires at least Python 3.6.

### What to install

Follow the [installation instructions](README.md#installation) to install the correct version of Python if you don't have it already.

I suggest using a powerful IDE such as [PyCharm](https://www.jetbrains.com/pycharm/download/download-thanks.html). [Visual Studio Code](https://code.visualstudio.com/) is also good with the right plugins. The nuad Python package uses type hints, and these tools are very helpful in giving static analysis warnings about the code that may represent errors that will manifest at run time.


### git

We use [git](https://git-scm.com/docs/gittutorial) and [GitHub](https://guides.github.com/activities/hello-world/). You can use the command-line git, or a GUI such as [GitHub desktop](https://desktop.github.com/), which is very easy to use and supports the most common git commands, but it is not fully-featured, so you may want another [git client](https://www.google.com/search?q=git+client) if you prefer not to use the command line.















## Cloning

The first step is cloning the repository so you have it available locally.

```
git clone https://github.com/UC-Davis-molecular-computing/nuad.git
```

Changes to the nuad package should be pushed to the
[`dev`](https://github.com/UC-Davis-molecular-computing/nuad/tree/dev) branch. So switch to the `dev` branch:

```
git checkout dev
```











## Pushing to the repository dev branch and documenting changes (done on all updates)

Minor changes, such as updating README, adding example files, etc., can be committed directly to the `dev` branch. (Note: currently this option is only available to administrators; other contributors should follow the instructions below.)

For any more significant change that is made (e.g., closing an issue, adding a new feature), follow these steps:

1. If there is not already a GitHub issue describing the desired change, make one. Make sure that its title is a self-contained description, and that it describes the change we would like to make to the software. For example, *"problem with importing gridless design"* is a bad title. A better title is *"fix problem where importing gridless design with negative x coordinates throws exception"*.

2. Make a new branch specifically for the issue. Base this branch off of `dev` (**WARNING**: in GitHub desktop, the default is to base it off of `main`, so switch that). The title of the issue (with appropriate hyphenation) is a good name for the branch. (In GitHub Desktop, if you paste the title of the issue, it automatically adds the hyphens.)

3. If it is about fixing a bug, *first* add tests to reproduce the bug before working on fixing it. (This is so-called [test-driven development](https://www.google.com/search?q=test-driven+development))

4. If it is about implementing a feature, first add tests to test the feature. For instance, if you are adding a new method, this involves writing code that calls the method and tests various combinations of example inputs and expected output.

5. Work entirely in that branch to fix the issue.

6. Run unit tests and ensure they pass.

7. Commit the changes. In the commit message, reference the issue using the phrase "fixes #123" or "closes #123" (see [here](https://docs.github.com/en/enterprise/2.16/user/github/managing-your-work-on-github/closing-issues-using-keywords)). Also, in the commit message, describe the issue that was fixed (one easy way is to copy the title of the issue); this message will show up in automatically generated release notes, so this is part of the official documentation of what changed.

8. Create a pull request (PR). **WARNING:** by default, it will want to merge into the `main` branch. Change the destination branch to `dev`.

9. Wait for all checks to complete (see next section), and then merge the changes from the new branch into `dev`. This will typically require someone else to review the code first and possibly request changes.

10. After merging, it will say that the branch you just merged from can be safely deleted. Delete the branch.

11. Locally, remember to switch back to the `dev` branch and pull it. (Although you added those changes locally, they revert back once you switch to your local `dev` branch, which needs to be synced with the remote repo for you to see the changes that were just merged from the now-deleted temporary branch.)









## Pushing to the repository main branch and documenting changes (done less frequently)

Less frequently, pull requests (abbreviated PR) can be made from `dev` to `main`, but make sure that `dev` is working before merging to `main`, since changes to the docstrings automatically update [readthedocs](https://nuad.readthedocs.io/en/latest/), which is the site hosting the API documentation. That is, changes to main immediately affect users reading online documentation, so it is critical that these work. Eventually we will automatically upload to PyPI, so this will also affect users installing via pip.

**WARNING:** Always wait for the checks to complete. This is important to ensure that unit tests pass. 

Although the GitHub web interface abbreviates long commit messages, the full commit message is included for each commit in a PR.

However, commit descriptions are not shown in the release notes. In GitHub desktop these are two separate fields; on the command line they appear to be indicated by two separate usages of the `-m` flag: https://stackoverflow.com/questions/16122234/how-to-commit-a-change-with-both-message-and-description-from-the-command-li.

So make sure that everything people should see in the automatically generated release notes is included in the commit message. (If not, then more manual editing of the release notes is required.) GitHub lets you [automatically close](https://docs.github.com/en/enterprise/2.16/user/github/managing-your-work-on-github/closing-issues-using-keywords) an issue by putting a phrase such as "closes #14". Although the release notes will link to the issue that was closed, they [will not describe it in any other way](https://github.com/marvinpinto/actions/issues/34). So it is important, for the sake of having readable release notes, to describe briefly the issue that was closed in the commit message.

One simple way to do this is to copy/paste the title of the issue into the commit message. For this reason, issue titles should be stated in terms of what change should happen to handle an issue. For example, instead of the title being *"3D position is improperly calculated from grid position"*, a better issue title is *"calculate 3D position correctly from grid position"*. That way, when the issue is fixed in a commit, that title can simply be copied and pasted as the description of what was done for the commit message. (But you should still add "fixes #<issue_number>" in the commit message, e.g., the full commit message could be *"fixes #101; calculate 3D position correctly from grid position"* .)

Users can read the description by clicking on the link to the commit or the pull request, but anything is put there, then the commit message should say something like "click on commit/PR for more details".

Breaking changes should be announced explicitly, perhaps in the commit message, but ideally also manually added at the top of the release notes, indicating what users need to do to deal with the change.

So the steps for committing to the main branch are:

1. If necessary, follow the instructions above to merge changes from a temporary branch to the `dev` branch. There will typically be several of these. Despite GitHub's suggestions to keep commit messages short and put longer text in descriptions, because only the commit message is included in the release notes, it's okay to put more detail in the message (but very long stuff should go in the description, or possibly documentation such as the README.md file).

3. Ensure all unit tests pass.

4. In the Python repo, ensure that the documentation is generated without errors. First, run `pip install sphinx sphinx_rtd_theme`. This installs [Sphinx](https://www.sphinx-doc.org/en/main/), which is the most well-supported documentation generator for Python. (It's not very friendly, the syntax for things like links in docstrings is awkward, but it's well supported, so we use it.) Then, from within the subfolder `doc`, run the command `make html` (or `make.bat html` on Windows), ensure there are no errors, and inspect the documentation it generates in the folder `_build`.

5. Create a PR to merge changes from dev into main. 

6. One the PR is reviewed and approved, do the merge.








## Styleguide

Follow the [Python style guide](https://www.python.org/dev/peps/pep-0008/), which should come along in most IDEs in the form of plugins and extensions. 

The line length should be configured to 110, as the style guide limit of 79 is a bit too restrictive.


## Miscellaneous

### Creating the NUPACK Docker Image

For future reference, I list out the steps to create the [Docker image for NUPACK](https://hub.docker.com/r/unhumbleben/nupack).

#### Prerequisite

* [Install NUPACK](http://www.nupack.org/downloads/register)
* [Install Docker](https://docs.docker.com/engine/install/)

#### Dockerfile

Create the following Dockerfile in the same directory where NUPACK is installed:
```
FROM ubuntu:20.04
COPY nupack-4.0.0.27/ /nupack
# Install ca-certificates to enable SSL
RUN apt-get update && apt-get install -y ca-certificates
```

The first line sets the [ubuntu:20.04](https://hub.docker.com/_/ubuntu) 
image as the base image from which we will build our NUPACK image.

The second line [copies](https://docs.docker.com/engine/reference/builder/#copy) new files or directories from the local
filesystem nupack-4.0.0.27/ and adds them to the filesystem of the container at the path /nupack. You may need to change the first argument
(nupack-4.0.0.27/) depending on the exact version of NUPACK you have.

The last line is needed to [enable SSL](https://packages.debian.org/sid/ca-certificates), which is required to run pip.

#### Build the Image
```
docker build -t unhumbleben/nupack .
```

The `-t unhumbleben/nupack` argument names the image
`unhumbleben/nupack`, but can be named to anything for testing purposes.

To play with the container, you can run

```
docker run -it <name-of-container>
```

To push the image to DockerHub, follow the instructions in the
[Docker documentation](https://docs.docker.com/get-started/04_sharing_app/#push-the-image).
