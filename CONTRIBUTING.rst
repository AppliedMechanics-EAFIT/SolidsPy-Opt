============
Contributing
============

Contributions to **SolidsPy-Opt** are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

Types of Contributions
----------------------

You can contribute in many ways:

Create Topology Optimization Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you run a topology optimization analysis (or general Finite Element Analysis)
using **SolidsPy-Opt**, and want to share it with the community, feel free
to submit a pull request or example to our repositories.

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/AppliedMechanics-EAFIT/SolidsPy-Opt/issues.

When reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* If possible, detailed steps to reproduce the bug.
* If the issue is hard to reproduce, please share any observations in as
  much detail as you can. Questions to start a discussion about the issue
  are welcome.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" is
open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with
"enhancement" and "please-help" is open to whoever wants to implement it.

Please do not combine multiple feature enhancements into a single pull request.

Write Documentation
~~~~~~~~~~~~~~~~~~~

SolidsPy-Opt could always use more documentation—whether as part of the official
docs, in docstrings, or even on the web in blog posts, articles, etc.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at
https://github.com/AppliedMechanics-EAFIT/SolidsPy-Opt/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Contributor Guidelines
----------------------

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

Before submitting a pull request, check that it meets these guidelines:

1. The pull request should include tests (if applicable).
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md (or wherever appropriate).
3. The pull request should work for Python 3. We do not officially support
   older versions of Python at this time.

Coding Standards
~~~~~~~~~~~~~~~~

* Adhere to PEP 8 style guidelines.
* Prefer functions over classes except in tests (where classes may be needed for organization).
* String quote conventions, per http://stackoverflow.com/a/56190/5549:
  
  - Use **double quotes** around strings that are used for interpolation or that are natural language messages.
  - Use **single quotes** for small symbol-like strings (but break the rules if the strings contain quotes).
  - Use **triple double quotes** for docstrings and raw string literals for regular expressions, even if they aren't strictly needed.
  - Example:

    .. code-block:: python

       LIGHT_MESSAGES = {
           'English': "There are %(number_of_lights)s lights.",
           'Pirate':  "Arr! Thar be %(number_of_lights)s lights."
       }

       def lights_message(language, number_of_lights):
           """Return a language-appropriate string reporting the light count."""
           return LIGHT_MESSAGES[language] % locals()

       def is_pirate(message):
           """Return True if the given message sounds piratical."""
           return re.search(r"(?i)(arr|avast|yohoho)!", message) is not None

* Write new code in Python 3.


