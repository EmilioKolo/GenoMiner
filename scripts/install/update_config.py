#!/usr/bin/env python3

import configparser
import sys
import os

# Command Line Argument Handling
if len(sys.argv) != 5:
    print(f"Usage: python3 {sys.argv[0]} <config_file> <section> <variable_name> <new_value>")
    print("Example: python3 update_config.py config.ini Parameters THREADS 4")
    sys.exit(1)

config_file_path = sys.argv[1]
section_name = sys.argv[2]
variable_name = sys.argv[3]
new_value = sys.argv[4]

if not os.path.exists(config_file_path):
    print(f"Error: Configuration file not found at '{config_file_path}'")
    sys.exit(1)

# Config Update Logic
try:
    # Initialize the parser
    config = configparser.ConfigParser()
    
    # Preserve the case of keys/options (e.g., 'BASE_DIR' vs 'base_dir')
    config.optionxform = str 
    
    # Read the existing config file
    config.read(config_file_path)

    # Check if the section exists, and create it if it does not
    if not config.has_section(section_name):
        config.add_section(section_name)
        print(f"Warning: Section '[{section_name}]' did not exist and was created.")
    
    # Set the new value
    config.set(section_name, variable_name, new_value)
    
    # Check if the variable was updated or newly added for better feedback
    action = "Updated" if variable_name in config[section_name] else "Added"
    
    print(f"Successfully {action.lower()} '{variable_name}' in section '[{section_name}]' to '{new_value}'")

    # Write the changes back to the config file
    with open(config_file_path, 'w') as configfile:
        config.write(configfile)

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
