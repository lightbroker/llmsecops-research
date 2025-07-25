import json
import random
from collections import defaultdict

def process_json_file(input_file, output_file, target_count=300):
    """
    Process JSON file to select requests from different source files.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        target_count (int): Target number of results (default: 300)
    """
    
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group data by source_file
    source_groups = defaultdict(list)
    for item in data:
        source_file = item.get('source_file', 'unknown')
        source_groups[source_file].append(item)
    
    print(f"Found {len(source_groups)} unique source files:")
    for source_file, items in source_groups.items():
        print(f"  {source_file}: {len(items)} items")
    
    # Select at least 1 item from each source file
    selected_items = []
    
    # First, pick one item from each source file
    for source_file, items in source_groups.items():
        selected_item = random.choice(items)
        selected_items.append(selected_item)
    
    print(f"\nSelected {len(selected_items)} items (1 from each source file)")
    
    # If we need more items to reach target_count, randomly select from all remaining items
    if len(selected_items) < target_count:
        # Create a pool of all items not yet selected
        remaining_items = []
        selected_set = set(id(item) for item in selected_items)
        
        for items in source_groups.values():
            for item in items:
                if id(item) not in selected_set:
                    remaining_items.append(item)
        
        # Randomly select additional items
        additional_needed = min(target_count - len(selected_items), len(remaining_items))
        additional_items = random.sample(remaining_items, additional_needed)
        selected_items.extend(additional_items)
    
    # If we have more than target_count, randomly trim to target_count
    if len(selected_items) > target_count:
        selected_items = random.sample(selected_items, target_count)
    
    print(f"Final selection: {len(selected_items)} items")
    
    # Process selected items: remove unwanted fields and add ID
    result = []
    for i, item in enumerate(selected_items, 1):
        processed_item = {
            'request': item['request'],
            'source_file': item['source_file'],
            'id': i
        }
        result.append(processed_item)
    
    # Shuffle the final result to randomize order
    random.shuffle(result)
    
    # Re-assign sequential IDs after shuffling
    for i, item in enumerate(result, 1):
        item['id'] = i
    
    # Write the result to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete! Results saved to {output_file}")
    
    # Print summary statistics
    source_counts = defaultdict(int)
    for item in result:
        source_counts[item['source_file']] += 1
    
    print(f"\nFinal distribution by source file:")
    for source_file, count in sorted(source_counts.items()):
        print(f"  {source_file}: {count} items")

def main():
    """Main function to run the script"""
    
    try:
        target_count = int(input("Enter target number of results (default: 300): ").strip() or "300")
    except ValueError:
        target_count = 300
        print("Invalid input, using default target count of 300")

    input_file = "./failed_entries.json"
    output_file = f"process_random_requests.output.{target_count}_count.json"
    
    try:
        process_json_file(input_file, output_file, target_count)
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{input_file}'")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()