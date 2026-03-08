import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "pinn_landslide"

list_of_files = [
    ".github/workflows/.gitkeep",
    # Source code directory
    f"src/{project_name}/__init__.py",
    
    # Components: The core building blocks of your methodology
    f"src/{project_name}/components/__init__.py",
      # Factor of Safety (FS) calculations
    
    # Utils: Helper functions (e.g., saving models, normalizing data)
    f"src/{project_name}/utils/__init__.py",
 
    # Configuration and Parameters Management
    f"src/{project_name}/config/__init__.py",
   
    # Pipeline: Connects the components together
    f"src/{project_name}/pipeline/__init__.py",
    
    
    # Entity: Custom data types and dataclasses (e.g., GeotechnicalParams)
    f"src/{project_name}/entity/__init__.py",

    # Constants: Hardcoded values that rarely change
    f"src/{project_name}/constants/__init__.py",
    
    # Custom Exception and Logger
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/exception/exception.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/logger/logger.py",
    
    # Config files for DVC, hyperparameters, and environment setup
    "config/config.yaml",              # File paths (e.g., path to DHM rainfall csv)
    "params.yaml",                     # Hyperparameters (learning rates, lambda weights, etc.)
                      # Data Version Control pipeline tracking
    "requirements.txt",
    "setup.py",

]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
        
    if (not os.path.exists(filepath) or (os.path.getsize(filepath) == 0)):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")