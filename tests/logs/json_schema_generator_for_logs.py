import json
import sys
from typing import Any, Dict, List, Union
from collections import defaultdict

class JSONSchemaGenerator:
    def __init__(self):
        self.type_counts = defaultdict(int)
    
    def get_python_type(self, value: Any) -> str:
        """Determine the Python type of a value."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"
    
    def analyze_array(self, arr: List[Any], path: str = "") -> Dict[str, Any]:
        """Analyze an array and determine the schema for its items."""
        if not arr:
            return {
                "type": "array",
                "items": {"type": "unknown"},
                "minItems": 0,
                "maxItems": 0
            }
        
        # Collect types and schemas of all items
        item_schemas = []
        type_frequency = defaultdict(int)
        
        for item in arr:
            item_type = self.get_python_type(item)
            type_frequency[item_type] += 1
            
            if item_type == "object":
                item_schemas.append(self.analyze_object(item, f"{path}[item]"))
            elif item_type == "array":
                item_schemas.append(self.analyze_array(item, f"{path}[item]"))
            else:
                item_schemas.append({"type": item_type})
        
        # Determine the most common type or create a union type
        most_common_type = max(type_frequency.items(), key=lambda x: x[1])[0]
        
        schema = {
            "type": "array",
            "minItems": len(arr),
            "maxItems": len(arr)
        }
        
        if len(type_frequency) == 1:
            # All items are the same type
            if most_common_type == "object" and item_schemas:
                # Merge object schemas
                schema["items"] = self.merge_object_schemas(item_schemas)
            elif most_common_type == "array" and item_schemas:
                # For arrays of arrays, use the first array's schema as template
                schema["items"] = item_schemas[0]
            else:
                schema["items"] = {"type": most_common_type}
        else:
            # Mixed types - create anyOf schema
            unique_schemas = []
            seen_schemas = set()
            
            for item_schema in item_schemas:
                schema_str = json.dumps(item_schema, sort_keys=True)
                if schema_str not in seen_schemas:
                    seen_schemas.add(schema_str)
                    unique_schemas.append(item_schema)
            
            schema["items"] = {
                "anyOf": unique_schemas
            }
        
        return schema
    
    def merge_object_schemas(self, schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple object schemas into one."""
        if not schemas:
            return {"type": "object"}
        
        merged = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Collect all properties
        all_properties = set()
        property_frequency = defaultdict(int)
        property_schemas = defaultdict(list)
        
        for schema in schemas:
            if "properties" in schema:
                for prop, prop_schema in schema["properties"].items():
                    all_properties.add(prop)
                    property_frequency[prop] += 1
                    property_schemas[prop].append(prop_schema)
        
        # Build merged properties
        total_schemas = len(schemas)
        for prop in all_properties:
            prop_count = property_frequency[prop]
            prop_schema_list = property_schemas[prop]
            
            # If property appears in all schemas, it's required
            if prop_count == total_schemas:
                merged["required"].append(prop)
            
            # Merge property schemas
            if len(set(json.dumps(s, sort_keys=True) for s in prop_schema_list)) == 1:
                # All schemas are identical
                merged["properties"][prop] = prop_schema_list[0]
            else:
                # Different schemas - create anyOf
                unique_schemas = []
                seen = set()
                for ps in prop_schema_list:
                    ps_str = json.dumps(ps, sort_keys=True)
                    if ps_str not in seen:
                        seen.add(ps_str)
                        unique_schemas.append(ps)
                
                if len(unique_schemas) == 1:
                    merged["properties"][prop] = unique_schemas[0]
                else:
                    merged["properties"][prop] = {"anyOf": unique_schemas}
        
        if not merged["required"]:
            del merged["required"]
        
        return merged
    
    def analyze_object(self, obj: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Analyze an object and generate its schema."""
        schema = {
            "type": "object",
            "properties": {},
            "required": list(obj.keys())
        }
        
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            value_type = self.get_python_type(value)
            
            if value_type == "object":
                schema["properties"][key] = self.analyze_object(value, current_path)
            elif value_type == "array":
                schema["properties"][key] = self.analyze_array(value, current_path)
            else:
                prop_schema = {"type": value_type}
                
                # Add additional constraints based on type
                if value_type == "string" and value:
                    prop_schema["minLength"] = len(value)
                    prop_schema["maxLength"] = len(value)
                elif value_type == "integer":
                    prop_schema["minimum"] = value
                    prop_schema["maximum"] = value
                elif value_type == "number":
                    prop_schema["minimum"] = value
                    prop_schema["maximum"] = value
                
                schema["properties"][key] = prop_schema
        
        return schema
    
    def generate_schema(self, data: Any) -> Dict[str, Any]:
        """Generate a JSON schema from the provided data."""
        root_type = self.get_python_type(data)
        
        if root_type == "object":
            return self.analyze_object(data)
        elif root_type == "array":
            return self.analyze_array(data)
        else:
            return {"type": root_type}
    
    def load_and_analyze(self, file_path: str) -> Dict[str, Any]:
        """Load a JSON file and generate its schema."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            schema = self.generate_schema(data)
            
            # Add schema metadata
            schema["$schema"] = "http://json-schema.org/draft-07/schema#"
            schema["title"] = f"Generated schema for {file_path}"
            schema["description"] = f"Auto-generated JSON schema based on the structure of {file_path}"
            
            return schema
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
        except Exception as e:
            raise Exception(f"Error processing file {file_path}: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python json_schema_generator.py <json_file_path>")
        print("Example: python json_schema_generator.py data.json")
        sys.exit(1)
    
    file_path = sys.argv[1]
    generator = JSONSchemaGenerator()
    
    try:
        schema = generator.load_and_analyze(file_path)
        
        # Pretty print the schema
        print(json.dumps(schema, indent=2, ensure_ascii=False))
        
        # Optionally save to file
        output_file = file_path.replace('.json', '_schema.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        
        print(f"\nSchema saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()