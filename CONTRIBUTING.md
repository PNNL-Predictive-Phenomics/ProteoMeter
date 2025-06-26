# WARNING: THIS IS OUT OF DATE SINCE WE HAVE CHANGED TOOLING


# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/PNNL-Predictive-Phenomics/ProteoMeter/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

ProteoMeter could always use more documentation, whether as part of the
official ProteoMeter docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/PNNL-Predictive-Phenomics/ProteoMeter/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.

## Get Started!

Ready to contribute? Here's how to set up `proteometer` for local development.

1. Fork the `proteometer` repo on GitHub.
2. Clone your fork locally

```
    $ git clone git@github.com:your_name_here/proteometer.git
```

3. Ensure [pixi](https://pixi.sh) is installed.
4. Install dependencies and start your virtualenv:

```
    $ pixi install
```

5. Create a branch for local development:

```
    $ git checkout -b name-of-your-bugfix-or-feature
```

   Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests, including linting and type-checking:

```
    $ pixi run all_checks
```

7. Commit your changes and push your branch to GitHub:

```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
```

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests if a new feature is added.
2. The pull request should include any necessary updates to docstrings.
