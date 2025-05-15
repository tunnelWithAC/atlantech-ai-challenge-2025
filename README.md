# Atlantech AI Challenge 2025

## Running the UI Code

To run the UI code locally:

1. Navigate to the UI directory:
   ```
   cd UI
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```

The UI will be available at http://localhost:5173 (or the port specified in the terminal output).

## UI Technologies

The UI is built with:
- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## Running the API code

### Python Environment Setup Guide

This guide explains how to set up a Python virtual environment and install packages from a requirements.txt file.

### Why Use Virtual Environments?

Virtual environments allow you to create isolated spaces for your Python projects, ensuring dependencies for different projects don't conflict with each other.

### Setting Up a Virtual Environment

#### Option 1: Using venv (Python 3.3+)

Python's built-in `venv` module is the standard way to create virtual environments.

```bash
# Navigate to your project directory
cd your-project-directory

# Create a virtual environment
python3 -m venv env

# On macOS/Linux:
source env/bin/activate

pip3 install -r requirements.txt
```

#### Option 2: Using virtualenv

If you prefer using `virtualenv` (especially for Python 2 compatibility):

```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create a virtual environment
virtualenv env

# Activate the virtual environment
# On Windows:
env\Scripts\activate

# On macOS/Linux:
source env/bin/activate
```

#### Option 3: Using Conda

If you're using Anaconda or Miniconda:

```bash
# Create a conda environment
conda create --name myenv python=3.9

# Activate the environment
conda activate myenv
```

### Installing Packages from requirements.txt

Once your virtual environment is activated (you should see the environment name in your terminal prompt), you can install packages from your requirements.txt file:

```bash
pip install -r requirements.txt
```

If you're using Conda:

```bash
conda install --file requirements.txt
```

### Deactivating the Virtual Environment

When you're done working on your project, you can deactivate the virtual environment:

```bash
# For venv and virtualenv
deactivate

# For Conda
conda deactivate
```

### Creating a requirements.txt File

If you need to create a requirements.txt file for your project:

```bash
pip freeze > requirements.txt
```

### Troubleshooting

- **Permission errors**: Try running commands with sudo (Linux/macOS) or administrator privileges (Windows)
- **Python not found**: Ensure Python is properly installed and in your PATH
- **Pip not found**: Ensure pip is installed and updated: `python -m pip install --upgrade pip`
- **Package installation failures**: Check internet connection and try updating pip

### Additional Resources

- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [virtualenv documentation](https://virtualenv.pypa.io/en/latest/)
- [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [pip documentation](https://pip.pypa.io/en/stable/) 

## Project Structure

- `/UI` - Frontend application
- `/api` - Backend API
- `/requirements.txt` - Python dependencies 

