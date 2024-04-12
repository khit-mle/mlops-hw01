#!/bin/bash

# Define the function to run Python scripts and check for errors
run_script() {
	script_name=$1
	echo "Running $script_name..."
	# Redirect stderr to stdout and capture output
	output=$(python "$script_name" 2>&1)
	local status=$?
	if [ $status -ne 0 ]; then
		echo "Error: $script_name failed with exit status $status."
		echo "Output:"
		echo "$output" # Display the output which includes both stdout and stderr
		exit $status
	else
		echo "$output" # Optionally display output on success as well
		echo "$script_name completed successfully."
	fi
}

# List of scripts to be run in order
scripts=(
	"data_creation.py"
	"model_preprocessing.py"
	"model_preparation.py"
	"model_testing.py"
)

# Loop through the scripts and run them
for script in "${scripts[@]}"; do
	run_script "$script"
done

echo "All scripts completed successfully."
