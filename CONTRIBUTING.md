# Contributing to Anote

First off, thank you for taking the time to contribute! The following is a set of guidelines for contributing to the open source ecosystem and supporting libraries hosted here. This is meant to help the review process go smoothly, save the reviewer(s) time in catching common issues, and avoid submitting PRs that will be rejected.

### How to Contribute?
If you want to contribute, start working through the Anote codebase and documentation to ensure that you can run the code. Follow the ```CODEBASE_SETUP.md``` file, and join our slack channel to ask any questions. Then, navigate to the issues tab, and pick one of the mentioned issues. Feel free to ask questions in the slack channel regarding any context on the issue. Create a separate branch, ideally named after the issue you are looking to solve. Feel free to create a new issue if that issue does not already exist (with the formal ticket name / number, description, assignee, priority, subtasks). Follow the pull request checklist below, and then submit a PR. One of our technical contributors / admins will review, and either merge the PR or provide comments.

### Pull-Request Checklist
The following is a list of tasks to be completed before submitting a pull request for final review.

#### Before creating PR:
- Follow coding best practices

- Make sure all new classes/functions/methods have docstrings.
   
- Make sure all new functions/methods have type hints (optional for tests).

#### Ensure environment is consistent
- Update dependencies in files if needed.
 
- Follow the virtualenv install instructions if you are unsure about working with virtual environments.

#### Ensure code is clean
- Remove all debugging artifacts.
  
- Remove commented out code.

- Remove any additional files / modifications that aren't core to the PR committed.
  
- For actual comments, note that our typical format is ```# TODO (<username>): <comment>```

- Double check everything has been committed and pushed, recommended that local feature branch is clean.

#### PR Guidelines:
- PR title should follow conventional commit standards (mention the issue number).

- PR description should give enough detail that the reviewer knows what they reviewing.

- If applicable, add a testing section to the PR description that recommends steps a reviewer can take to verify the changes, e.g. a snippet of code they can run locally.

#### License
Anote open source projects are licensed under the Apache 2.0 license.

#### Conventions
For pull requests, our convention is to squash and merge. For PR titles, we use conventional commit messages. The format should look like

```<type>: <description>```.
For example, if the PR addresses a new feature, the PR title should look like:

```feat: Implements exciting new feature```.
For feature branches, the naming convention is:

```<username>/<description>```.
For the commit above, coming from the user called contributor the branch name would look like:

```contributor/exciting-new-feature```.
Here is a list of some of the most common possible commit types:


Below are some terms that can be helpful when submitting a pull request:

| **Type**    | **Description**                                                                 |
|-------------|---------------------------------------------------------------------------------|
| **feat**    | A new feature is introduced with the changes                                   |
| **fix**     | A bug fix has occurred                                                         |
| **chore**   | Changes that do not relate to a fix or feature (e.g., updating dependencies) |
| **refactor**| Refactored code that neither fixes a bug nor adds a feature                    |
| **docs**    | Updates to documentation such as the README or other markdown files            |

#### Why should you write better commit messages?
By writing good commits, you are simply future-proofing yourself. You could save yourself and/or coworkers hours of digging around while troubleshooting by providing that helpful description. The extra time it takes to write a thoughtful commit message as a letter to your potential future self is extremely worthwhile. On large scale projects, documentation is imperative for maintenance. Collaboration and communication are of utmost importance within engineering teams. The Git commit message is a prime example of this. I highly suggest setting up a convention for commit messages on your team if you do not already have one in place.

#### Code of Conduct
In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

#### Enforcement
Please report unacceptable behavior to nvidra@anote.ai. All complaints will be reviewed and investigated and will result in a response that is deemed necessary and appropriate to the circumstances. The project team is obligated to maintain confidentiality with regard to the reporter of an incident. Further details of specific enforcement policies may be posted separately. Project maintainers who do not follow or enforce the Code of Conduct in good faith may face temporary or permanent repercussions as determined by other members of the project's leadership.

Warm Regards,

The Anote Team
