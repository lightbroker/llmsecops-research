import json
import sys
from typing import Any, Dict, List, Set

class JSONKeyHierarchy:
    def __init__(self):
        self.key_paths = set()
    
    def extract_paths(self, data: Any, current_path: str = "") -> None:
        """Recursively extract all key paths from JSON data."""
        if isinstance(data, dict):
            for key, value in data.items():
                # Create the path for this key
                if current_path:
                    new_path = f"{current_path}['{key}']"
                else:
                    new_path = f"['{key}']"
                
                # Add this path to our collection
                self.key_paths.add(new_path)
                
                # Recursively process the value
                self.extract_paths(value, new_path)
                
        elif isinstance(data, list):
            # For arrays, we show the index access pattern and analyze first item
            if data:  # Only if array is not empty
                index_path = f"{current_path}[0]" if current_path else "[0]"
                self.key_paths.add(f"{current_path}[index]" if current_path else "[index]")
                
                # Analyze the structure of array items (using first item as template)
                self.extract_paths(data[0], f"{current_path}[index]" if current_path else "[index]")
                
                # If there are multiple items, check if they have different structures
                if len(data) > 1:
                    for i, item in enumerate(data[1:], 1):
                        temp_hierarchy = JSONKeyHierarchy()
                        temp_hierarchy.extract_paths(item, f"{current_path}[index]" if current_path else "[index]")
                        # Merge any new paths found
                        self.key_paths.update(temp_hierarchy.key_paths)
    
    def get_organized_paths(self) -> Dict[str, List[str]]:
        """Organize paths by depth and return a structured view."""
        organized = {
            "root_keys": [],
            "nested_paths": [],
            "array_paths": []
        }
        
        for path in sorted(self.key_paths):
            if path.count('[') == 1 and '[index]' not in path:
                # Root level keys
                organized["root_keys"].append(path)
            elif '[index]' in path:
                # Array-related paths
                organized["array_paths"].append(path)
            else:
                # Nested object paths
                organized["nested_paths"].append(path)
        
        return organized
    
    def print_hierarchy(self) -> None:
        """Print a formatted hierarchy of all key paths."""
        organized = self.get_organized_paths()
        
        print("JSON KEY HIERARCHY")
        print("=" * 50)
        print()
        
        if organized["root_keys"]:
            print("ROOT LEVEL KEYS:")
            print("-" * 20)
            for path in organized["root_keys"]:
                clean_key = path.strip("[]'")
                print(f"  {clean_key}")
                print(f"    Access: data{path}")
            print()
        
        if organized["nested_paths"]:
            print("NESTED OBJECT PATHS:")
            print("-" * 25)
            for path in organized["nested_paths"]:
                # Create a readable description
                parts = path.replace("']['", " → ").replace("['", "").replace("']", "").split(" → ")
                readable = " → ".join(parts)
                print(f"  {readable}")
                print(f"    Access: data{path}")
            print()
        
        if organized["array_paths"]:
            print("ARRAY PATHS:")
            print("-" * 15)
            print("  Note: Replace 'index' with actual array index (0, 1, 2, etc.)")
            print()
            for path in organized["array_paths"]:
                # Create a readable description
                readable_path = path.replace("']['", " → ").replace("['", "").replace("']", "").replace("[index]", "[i]")
                readable = readable_path.replace(" → ", " → ")
                print(f"  {readable}")
                print(f"    Access: data{path.replace('index', 'i')}")
            print()
    
    def generate_access_examples(self, sample_data: Any) -> None:
        """Generate example code showing how to access each path."""
        print("PYTHON ACCESS EXAMPLES:")
        print("=" * 30)
        print()
        
        organized = self.get_organized_paths()
        
        print("# Load your JSON data")
        print("import json")
        print("with open('your_file.json', 'r') as f:")
        print("    data = json.load(f)")
        print()
        
        if organized["root_keys"]:
            print("# Accessing root level keys:")
            for path in organized["root_keys"]:
                key = path.strip("[]'")
                print(f"{key}_value = data{path}")
            print()
        
        if organized["nested_paths"]:
            print("# Accessing nested keys:")
            for path in organized["nested_paths"]:
                var_name = path.replace("']['", "_").replace("['", "").replace("']", "").replace("-", "_").lower()
                print(f"{var_name} = data{path}")
            print()
        
        if organized["array_paths"]:
            print("# Accessing array elements:")
            print("# Replace 'index' with the actual index you want (0, 1, 2, etc.)")
            for path in organized["array_paths"]:
                if path.count('[index]') == 1:  # Only show simple array paths
                    var_name = path.replace("']['", "_").replace("['", "").replace("']", "").replace("[index]", "_item").replace("-", "_").lower()
                    example_path = path.replace("index", "0")  # Show example with index 0
                    print(f"# Example: {var_name} = data{example_path}")
                    print(f"# General: for i, item in enumerate(data{path.split('[index]')[0] if '[index]' in path else ''}): ...")
            print()
    
    def save_to_file(self, output_file: str) -> None:
        """Save the hierarchy to a text file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Redirect print output to file
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            
            self.print_hierarchy()
            self.generate_access_examples(None)
            
            sys.stdout = old_stdout
    
    def analyze_file(self, file_path: str) -> None:
        """Load and analyze a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Extract all paths
            self.extract_paths(data)
            
            # Print the hierarchy
            self.print_hierarchy()
            
            # Generate access examples
            self.generate_access_examples(data)
            
            # Save to file
            output_file = file_path.replace('.json', '_key_paths.txt')
            self.save_to_file(output_file)
            print(f"Key hierarchy saved to: {output_file}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
        except Exception as e:
            raise Exception(f"Error processing file {file_path}: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python json_key_hierarchy.py <json_file_path>")
        print("Example: python json_key_hierarchy.py data.json")
        print()
        print("This script will:")
        print("- Show all key paths in your JSON file")
        print("- Provide Python code examples for accessing each key")
        print("- Save the results to a text file")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyzer = JSONKeyHierarchy()
    
    try:
        analyzer.analyze_file(file_path)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()