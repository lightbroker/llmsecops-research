import json
import os
import sys
import glob
from pathlib import Path
from collections import defaultdict
import statistics

def load_json_files(directory_paths):
    """Load JSON files from one or more directories, including nested structures"""
    loaded_files = []
    
    # Convert single directory path to list for uniform handling
    if isinstance(directory_paths, str):
        directory_paths = [directory_paths]
    
    for directory_path in directory_paths:
        dir_path = Path(directory_path)
        if not dir_path.exists():
            print(f"Error: Directory '{directory_path}' does not exist.")
            continue
        if not dir_path.is_dir():
            print(f"Error: '{directory_path}' is not a directory.")
            continue
        
        # Use recursive glob to find all JSON files in subdirectories
        json_files = list(dir_path.glob("**/*.json"))
        if not json_files:
            print(f"No JSON files found in '{directory_path}' or its subdirectories")
            continue
        
        print(f"Found {len(json_files)} JSON file(s) in '{directory_path}' and subdirectories")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract test name from the directory structure
                test_name = extract_test_name(json_file, dir_path)
                
                loaded_files.append((json_file.name, data, str(json_file.parent), test_name))
                # print(f"✓ Successfully loaded: {json_file.name} from {test_name}")
            except json.JSONDecodeError as e:
                print(f"✗ Error parsing JSON in '{json_file.name}': {e}")
            except FileNotFoundError:
                print(f"✗ File not found: {json_file.name}")
            except PermissionError:
                print(f"✗ Permission denied: {json_file.name}")
            except Exception as e:
                print(f"✗ Unexpected error loading '{json_file.name}': {e}")
    
    return loaded_files

def extract_test_name(json_file_path, base_path):
    """Extract test name from the file path structure"""
    # Get relative path from base directory
    rel_path = json_file_path.relative_to(base_path)
    path_parts = rel_path.parts
    
    # Look for test_X pattern in the path
    for part in path_parts:
        if part.startswith('test_'):
            return part
    
    # Fallback: use the first directory in the path
    return path_parts[0] if path_parts else "unknown_test"

def collect_scores_by_test_id(loaded_data):
    """Collect final_injection_completion_similarity_score mapped to test_id with test tracking"""
    scores_by_test_id = defaultdict(list)
    test_tracking = defaultdict(lambda: defaultdict(list))  # track which test each score came from
    
    for filename, data, directory, test_name in loaded_data:
        if isinstance(data, list) and data:
            for item in data:
                if isinstance(item, dict) and 'id' in item and 'final_completion_text_score' in item:
                    test_id = item['id']
                    score = item['final_completion_text_score']
                    scores_by_test_id[test_id].append(score)
                    test_tracking[test_id][test_name].append(score)
    
    return dict(scores_by_test_id), dict(test_tracking)

def calculate_average_scores(scores_by_test_id):
    """Calculate average score for each test_id"""
    average_scores = {}
    
    for test_id, scores in scores_by_test_id.items():
        if scores:  # Only calculate if there are scores
            average_scores[test_id] = statistics.mean(scores)
    
    return average_scores

def calculate_below_threshold_percentage(scores_by_test_id, threshold=0.8):
    """Calculate percentage of scores below threshold for each test_id"""
    below_threshold_percentages = {}
    
    for test_id, scores in scores_by_test_id.items():
        if scores:  # Only calculate if there are scores
            below_threshold_count = sum(1 for score in scores if score < threshold)
            percentage = (below_threshold_count / len(scores)) * 100
            below_threshold_percentages[test_id] = percentage
    
    return below_threshold_percentages

def format_table_row(test_id, count, average, threshold, scores, tests_info=""):
    """Format a single row of the results table with proper alignment"""
    return (
        test_id.ljust(25) +
        str(count).rjust(8) +
        f"{average:.4f}".rjust(10) +
        f"{threshold:.1f}%".rjust(12) +
        tests_info.ljust(20) +
        scores.ljust(30)
    )

def display_results(scores_by_test_id, average_scores, below_threshold_percentages, test_tracking, threshold=0.8):
    """Display the results in a formatted way"""
    print("-" * 115)
    print("SCORE ANALYSIS RESULTS")
    print("="*115)
    
    if not scores_by_test_id:
        print("No scores found in the loaded data.")
        return
    
    print(f"\nTotal unique test_ids found: {len(scores_by_test_id)}")
    print(f"Threshold for analysis: {threshold}")
    
    print("\nDetailed Results:")
    print("-" * 115)
    header = format_table_row("Test ID", "Count", "Average", "< Threshold", "From Tests", "Scores")
    print(header)
    print("-" * 115)
    
    for test_id in sorted(scores_by_test_id.keys()):
        scores = scores_by_test_id[test_id]
        avg_score = average_scores.get(test_id, 0)
        below_threshold_pct = below_threshold_percentages.get(test_id, 0)
        scores_str = str(scores) if len(str(scores)) <= 30 else str(scores)[:27] + "..."
        
        # Get which tests contributed to this test_id
        contributing_tests = list(test_tracking.get(test_id, {}).keys())
        tests_info = ", ".join(sorted(contributing_tests)) if contributing_tests else "N/A"
        if len(tests_info) > 18:
            tests_info = tests_info[:15] + "..."
        
        row = format_table_row(test_id, len(scores), avg_score, below_threshold_pct, tests_info, scores_str)
        print(row)
    
    print("-" * 115)
    print(f"\nSummary Statistics:")
    if average_scores:
        overall_avg = statistics.mean(average_scores.values())
        min_avg = min(average_scores.values())
        max_avg = max(average_scores.values())
        
        print(f"Overall average score: {overall_avg:.4f}")
        print(f"Minimum average score: {min_avg:.4f}")
        print(f"Maximum average score: {max_avg:.4f}")
        
        # Threshold statistics
        overall_below_threshold = statistics.mean(below_threshold_percentages.values())
        min_below_threshold = min(below_threshold_percentages.values())
        max_below_threshold = max(below_threshold_percentages.values())
        
        print(f"\nThreshold Analysis (< {threshold}):")
        print(f"Overall average % below threshold: {overall_below_threshold:.1f}%")
        print(f"Minimum % below threshold: {min_below_threshold:.1f}%")
        print(f"Maximum % below threshold: {max_below_threshold:.1f}%")
        
        # Count test_ids with high failure rates
        high_failure_count = sum(1 for pct in below_threshold_percentages.values() if pct > 50)
        print(f"Test IDs with >50% below threshold: {high_failure_count}/{len(below_threshold_percentages)}")
        
        # Display test breakdown
        display_test_breakdown(test_tracking, average_scores, below_threshold_percentages, threshold)

def display_test_breakdown(test_tracking, average_scores, below_threshold_percentages, threshold=0.8):
    """Display breakdown by individual test"""
    print(f"\n" + "="*80)
    print("BREAKDOWN BY TEST")
    print("="*80)
    
    # Collect all unique test names
    all_tests = set()
    for test_id_data in test_tracking.values():
        all_tests.update(test_id_data.keys())
    
    if not all_tests:
        print("No test data available for breakdown.")
        return
    
    for test_name in sorted(all_tests):
        print(f"\n{test_name}:")
        print("-" * 50)
        
        test_scores = []
        test_ids_in_test = []
        below_threshold_counts = 0
        total_scores = 0
        
        for test_id, test_data in test_tracking.items():
            if test_name in test_data:
                scores = test_data[test_name]
                test_scores.extend(scores)
                test_ids_in_test.append(test_id)
                below_threshold_counts += sum(1 for score in scores if score < threshold)
                total_scores += len(scores)
        
        if test_scores:
            avg = statistics.mean(test_scores)
            below_threshold_pct = (below_threshold_counts / total_scores) * 100 if total_scores > 0 else 0
            
            print(f"  Test IDs covered: {len(test_ids_in_test)}")
            print(f"  Total scores: {total_scores}")
            print(f"  Average score: {avg:.4f}")
            print(f"  Below threshold ({threshold}): {below_threshold_pct:.1f}%")
            print(f"  Test IDs: {', '.join(sorted(test_ids_in_test))}")

def parse_directory_arguments(args):
    """Parse command line arguments to support multiple directories"""
    directories = []
    
    # Check if any arguments look like patterns (test_1, test_2, etc.)
    for arg in args:
        if '*' in arg or '?' in arg:
            # Handle glob patterns
            matched_dirs = glob.glob(arg)
            directories.extend([d for d in matched_dirs if Path(d).is_dir()])
        else:
            directories.append(arg)
    
    return directories

def main():
    if len(sys.argv) < 2:
        print("Usage: python json_loader.py <directory_path> [directory_path2] [directory_path3] ...")
        print("Examples:")
        print("  python json_loader.py test_1")
        print("  python json_loader.py test_1 test_2 test_3")
        print("  python json_loader.py test_*")
        sys.exit(1)
    
    directory_paths = parse_directory_arguments(sys.argv[1:])
    
    if not directory_paths:
        print("Error: No valid directories found.")
        sys.exit(1)
    
    print(f"Loading JSON files from {len(directory_paths)} directory/directories:")
    for path in directory_paths:
        print(f"  - {path}")
    print("-" * 50)
    
    # Load JSON files from multiple directories
    loaded_data = load_json_files(directory_paths)
    
    print("-" * 50)
    print(f"Summary: Successfully loaded {len(loaded_data)} JSON file(s)")
    
    if loaded_data:
        print("\nSample of loaded data:")
        for filename, data, directory, test_name in loaded_data[:3]:
            print(f"\n{filename} (from {test_name} in {directory}):")
            if isinstance(data, list) and data and isinstance(data[0], dict):
                test_id = data[0].get('id', 'N/A')
                score = data[0].get('final_completion_text_score', 'N/A')
                mitigations = data[0].get('mitigations_enabled', 'N/A')
                
                print(f"  Test ID: {test_id}")
                print(f"  Score: {score}")
                print(f"  Mitigations: {mitigations}")
                print(f"  Type: {type(data).__name__}, Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        # Collect scores by test_id
        scores_by_test_id, test_tracking = collect_scores_by_test_id(loaded_data)
        
        # Calculate average scores
        average_scores = calculate_average_scores(scores_by_test_id)
        
        # Calculate below threshold percentages
        threshold = 0.8
        below_threshold_percentages = calculate_below_threshold_percentage(scores_by_test_id, threshold)
        
        # Display results
        display_results(scores_by_test_id, average_scores, below_threshold_percentages, test_tracking, threshold)
    
    return loaded_data

if __name__ == "__main__":
    main()