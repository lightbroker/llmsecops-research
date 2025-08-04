#!/usr/bin/env python3
"""
Script to filter JSON files for entries with "failed": true and combine them
into a single output file with source filename information.
"""

import json
import os
import glob
from pathlib import Path

def process_json_files(input_directory=".", output_file="failed_entries.json", pattern="*.json"):
    """
    Process all JSON files in a directory, filter for failed entries,
    and combine them into a single output file.
    
    Args:
        input_directory (str): Directory containing JSON files to process
        output_file (str): Name of the output file for failed entries
        pattern (str): File pattern to match (default: "*.json")
    """
    failed_entries = []
    processed_files = 0
    total_failed = 0
    
    # Get all JSON files matching the pattern
    json_files = glob.glob(os.path.join(input_directory, pattern))
    
    if not json_files:
        print(f"No JSON files found in '{input_directory}' matching pattern '{pattern}'")
        return
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both list of objects and single object
            if isinstance(data, list):
                objects_to_check = data
            else:
                objects_to_check = [data]
            
            # Filter for failed entries
            file_failed_count = 0
            for obj in objects_to_check:
                if isinstance(obj, dict) and obj.get("failed") is True:
                    # Add source filename to the object
                    obj_with_source = obj.copy()
                    obj_with_source["source_file"] = filename
                    failed_entries.append(obj_with_source)
                    file_failed_count += 1
            
            if file_failed_count > 0:
                print(f"  Found {file_failed_count} failed entries")
                total_failed += file_failed_count
            else:
                print(f"  No failed entries found")
            
            processed_files += 1
            
        except json.JSONDecodeError as e:
            print(f"  Error: Invalid JSON in {filename} - {e}")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    # Write the combined failed entries to output file
    if failed_entries:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(failed_entries, f, indent=2, ensure_ascii=False)
            
            print(f"\nSuccess!")
            print(f"Processed {processed_files} files")
            print(f"Found {total_failed} failed entries")
            print(f"Output written to: {output_file}")
            
        except Exception as e:
            print(f"Error writing output file: {e}")
    else:
        print(f"\nNo failed entries found in any of the {processed_files} processed files.")

def main():
    """
    Main function with configurable parameters.
    Modify these variables as needed for your specific use case.
    """
    # Configuration - modify these as needed
    INPUT_DIRECTORY = "."  # Current directory, change to your JSON files location
    OUTPUT_FILE = "failed_entries.json"  # Output file name
    FILE_PATTERN = "*_failures.json"  # Pattern to match JSON files
    
    print("JSON Failed Entries Filter")
    print("=" * 30)
    print(f"Input directory: {INPUT_DIRECTORY}")
    print(f"File pattern: {FILE_PATTERN}")
    print(f"Output file: {OUTPUT_FILE}")
    print()
    
    # Process the files
    process_json_files(INPUT_DIRECTORY, OUTPUT_FILE, FILE_PATTERN)

if __name__ == "__main__":
    main()