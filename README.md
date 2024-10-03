# Viceral-Fat

Author: Kanghyun Ryu
A medical image segmentation project.

## Overview

This project was generated using a Cookiecutter template. It provides a standardized structure for Research projects, making it easier to start and maintain consistent development practices.

## Features

- [List key features of the project template]
- [Highlight any specific tools or libraries included]
- [Mention any configuration or setup that's pre-done]

## Getting Started

### 1. Pipenv - Handling Dependencies and Libraries (Don't use Conda)

1. Install Pipenv (if not already installed):
   ```
   pip install pipenv
   ```

2. Clone the project and navigate to the directory:
   ```
   git clone [repository_url]
   cd viceral_fat
   ```

3. Create Pipenv environment and install dependencies:
   ```
   pipenv install
   ```

4. Activate the Pipenv environment:
   ```
   pipenv shell
   ```

5. Run Jupyter Notebook (if needed):
   ```
   pipenv run jupyter notebook
   ```

### 2. DVC - handling data management and tracking

1. Initialize `DVC` (skip if already initialized in the repository):
   ```
   dvc init
   ```

2. Add remote storage:
   ```
   dvc remote add -d myremote ssh://user@host:/path/to/dvc-backup
   ```

3. Modify remote storage settings:
   ```
   dvc remote modify myremote user username
   dvc remote modify myremote password password
   ```

4. Track data (skip if already added in the repository):
   ```
   dvc add data/dataset_name
   ```

5. Commit changes (skip if already committed in the repository):
   ```
   git add .dvc
   git commit -m "Add data to DVC"
   ```

6. Push data (skip for pull-only repositories):
   ```
   dvc push
   ```

7. Pull data (execute this step):
   ```
   dvc pull
   ```

8. Check data status:
   ```
   dvc status
   ```

9. Retrieve a specific version of data:
   ```
   git checkout <commit-hash>
   dvc checkout
   ```

## Project Structure

- `data/`: Directory for storing datasets
- `nbs/`: Directory for Jupyter Notebook files
- `script/`: Directory for experiment and test scripts
  - `unit_test/`: Directory for unit test scripts
- `src/`: Directory for source code
  - `archs/`: Code related to model architectures
  - `data/`: Code for data processing
  - `losses/`: Code for loss functions
  - `metrics/`: Code for evaluation metrics
  - `models/`: Code for model implementations
  - `utils/`: Directory for utility functions
- `.project-root`: use `autorootcwd` to automatically get root folder path
- `Pipfile`: use `pipenv` to handle and install dependencies

