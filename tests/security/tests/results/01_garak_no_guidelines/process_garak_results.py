#!/usr/bin/env python3
"""
Script to process NVIDIA Garak LLM vulnerability scanner results.
Matches failed responses from text file with JSON request/response pairs
and adds "failed": true flag to matching entries.
"""

import json
import re
from typing import List, Dict, Any


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by handling encoding differences.
    - Replace \n with spaces
    - Normalize whitespace (multiple spaces become single space)
    - Strip leading/trailing whitespace
    """
    # Replace literal \n with spaces
    normalized = text.replace('\\n', ' ')
    # Replace actual newlines with spaces
    normalized = normalized.replace('\n', ' ')
    # Normalize multiple spaces to single space
    normalized = re.sub(r'\s+', ' ', normalized)
    # Strip leading/trailing whitespace
    return normalized.strip()


def extract_failed_responses(text_file_path: str) -> List[str]:
    """
    Extract failed responses from the text file.
    Failed responses are marked with ❌ at the beginning.
    """
    failed_responses = []
    
    with open(text_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by ❌ to get individual failed responses
    parts = content.split('❌')
    
    for part in parts[1:]:  # Skip first empty part
        # Clean up the response text
        response = part.strip()
        if response:
            # Remove any trailing content after the response
            # Look for patterns that indicate end of response
            end_patterns = [
                r'\n📜',  # Log entries
                r'\n✔️',  # Success indicators
                r'\n❌',  # Next failure
            ]
            
            for pattern in end_patterns:
                match = re.search(pattern, response)
                if match:
                    response = response[:match.start()]
                    break
            
            failed_responses.append(normalize_text(response))
    
    return failed_responses


def load_json_data(json_file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file. Handle both single object and array formats.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Try to parse as JSON array first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        else:
            return [data]
    except json.JSONDecodeError:
        # If that fails, try parsing line by line (JSONL format)
        data = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data


def find_matching_responses(json_data: List[Dict[str, Any]], 
                          failed_responses: List[str]) -> List[Dict[str, Any]]:
    """
    Find JSON entries that match failed responses and mark them as failed.
    """
    processed_data = []
    
    for entry in json_data:
        # Create a copy of the entry
        processed_entry = entry.copy()
        
        if 'response' in entry:
            normalized_response = normalize_text(entry['response'])
            
            # Check if this response matches any failed response
            is_failed = False
            for failed_response in failed_responses:
                # Use fuzzy matching - check if the failed response is contained
                # in the JSON response or vice versa (accounting for truncation)
                if (failed_response in normalized_response or 
                    normalized_response in failed_response or
                    # Also check with high similarity threshold
                    calculate_similarity(normalized_response, failed_response) > 0.9):
                    is_failed = True
                    break
            
            processed_entry['failed'] = is_failed
        else:
            # If no response field, mark as not failed
            processed_entry['failed'] = False
        
        processed_data.append(processed_entry)
    
    return processed_data


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple similarity between two texts based on common words.
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def save_processed_data(data: List[Dict[str, Any]], output_file_path: str):
    """
    Save processed data to JSON file.
    """
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_summary(data: List[Dict[str, Any]]):
    """
    Print summary of processing results.
    """
    total_entries = len(data)
    failed_entries = sum(1 for entry in data if entry.get('failed', False))
    
    print(f"\nProcessing Summary:")
    print(f"Total entries: {total_entries}")
    print(f"Failed entries: {failed_entries}")
    print(f"Success rate: {((total_entries - failed_entries) / total_entries * 100):.1f}%")
    
    # Show some examples of failed entries
    failed_examples = [entry for entry in data if entry.get('failed', False)][:3]
    if failed_examples:
        print(f"\nFirst few failed entries:")
        for i, entry in enumerate(failed_examples, 1):
            response_preview = entry.get('response', '')[:100] + '...' if len(entry.get('response', '')) > 100 else entry.get('response', '')
            print(f"{i}. {response_preview}")


def generate_output_filename(json_file_path: str) -> str:
    """
    Generate output filename based on the JSON file path.
    Example: 'results_batch1.json' -> 'results_batch1_failures.json'
    """
    import os
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    return f"{base_name}_failures.json"


def process_file_pair(text_file_path: str, json_file_path: str) -> bool:
    """
    Process a single text/JSON file pair.
    Returns True if successful, False otherwise.
    """
    output_file_path = generate_output_filename(json_file_path)
    
    print(f"\nProcessing pair:")
    print(f"  Text file: {text_file_path}")
    print(f"  JSON file: {json_file_path}")
    print(f"  Output file: {output_file_path}")
    
    try:
        # Extract failed responses from text file
        print("  Extracting failed responses...")
        failed_responses = extract_failed_responses(text_file_path)
        print(f"  Found {len(failed_responses)} failed responses")
        
        # Load JSON data
        print("  Loading JSON data...")
        json_data = load_json_data(json_file_path)
        print(f"  Loaded {len(json_data)} JSON entries")
        
        # Match and mark failed entries
        print("  Matching responses and marking failures...")
        processed_data = find_matching_responses(json_data, failed_responses)
        
        # Save processed data
        print(f"  Saving processed data to {output_file_path}...")
        save_processed_data(processed_data, output_file_path)
        
        # Print summary
        print_summary(processed_data)
        
        print(f"  ✓ Processing complete! Results saved to {output_file_path}")
        return True
        
    except FileNotFoundError as e:
        print(f"  ✗ Error: File not found - {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error processing files: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to process multiple file pairs.
    """
    # Dictionary mapping text files to JSON files based on your directory structure
    # Format: 'text_log_file.txt': 'corresponding_http_log_file.json'
    file_mapping = {
        # DAN (Do Anything Now) attacks
        './garak_logs__dan.DanInTheWild.txt': './http/http_logs_1753209001__dan.DanInTheWild.json',
        './garak_logs__dan.DAN_Jailbreak.txt': './http/http_logs_1752960264__dan.DAN_Jailbreak.json',
        './garak_logs__dan.DUDE.txt': './http/http_logs_1753229565__dan.DUDE.json',
        './garak_logs__dan.STAN.txt': './http/http_logs_1752952243__dan.STAN.json',
        
        # ANSI Escape attacks
        './garak_logs__ansiescape.AnsiEscaped.txt': './http/http_logs_1752952966__ansiescape.AnsiEscaped.json',
        
        # Divergence attacks
        './garak_logs__divergence.Repeat.txt': './http/http_logs_1752960940__divergence.Repeat.json',
        
        # Encoding attacks
        './garak_logs__encoding.InjectAscii85.txt': './http/http_logs_1752963607__encoding.InjectAscii85.json',
        
        # SQL Injection attacks
        './garak_logs__exploitation.SQLInjectionEcho.txt': './http/http_logs_1752970863__exploitation.SQLInjectionEcho.json',
        './garak_logs__exploitation.SQLInjectionSystem.txt': './http/http_logs_1752971849__exploitation.SQLInjectionSystem.json',
        
        # Goodside attacks
        './garak_logs__goodside.Davidjl.txt': './http/http_logs_1752973184__goodside.Davidjl.json',
        './garak_logs__goodside.Tag.txt': './http/http_logs_1752973549__goodside.Tag.json',
        
        # Grandma attacks
        './garak_logs__grandma.Substances.txt': './http/http_logs_1752977387__grandma.Substances.json',
        './garak_logs__grandma.Win10.txt': './http/http_logs_1752943481__grandma.Win10.json',
        './garak_logs__grandma.Win11.txt': './http/http_logs_1753013213__grandma.Win11.json',
        
        # Latent Injection attacks
        './garak_logs__latentinjection.LatentInjectionFactSnippetEiffel.txt': './http/http_logs_1753014264__latentinjection.LatentInjectionFactSnippetEiffel.json',
        './garak_logs__latentinjection.LatentInjectionFactSnippetLegal.txt': './http/http_logs_1753146682__latentinjection.LatentInjectionFactSnippetLegal.json',
        './garak_logs__latentinjection.LatentInjectionReport.txt': './http/http_logs_1753231480__latentinjection.LatentInjectionReport.json',
        './garak_logs__latentinjection.LatentInjectionResume.txt': './http/http_logs_1753269270__latentinjection.LatentInjectionResume.json',
        './garak_logs__latentinjection.LatentJailbreak.txt': './http/http_logs_1753102185__latentinjection.LatentJailbreak.json',
        './garak_logs__latentinjection.LatentWhois.txt': './http/http_logs_1753020445__latentinjection.LatentWhois.json',
        
        # Leak Replay attacks
        './garak_logs__leakreplay.GuardianCloze.txt': './http/http_logs_1753014376__leakreplay.GuardianCloze.json',
        './garak_logs__leakreplay.GuardianComplete.txt': './http/http_logs_1753018304__leakreplay.GuardianComplete.json',
        './garak_logs__leakreplay.NYTComplete.txt': './http/http_logs_1753124308__leakreplay.NYTComplete.json',
        
        # Malware Generation attacks
        './garak_logs__malwaregen.Evasion.txt': './http/http_logs_1753040768__malwaregen.Evasion.json',
        './garak_logs__malwaregen.Payload.txt': './http/http_logs_1753095905__malwaregen.Payload.json',
        
        # Misleading attacks
        './garak_logs__misleading.FalseAssertion.txt': './http/http_logs_1752944788__misleading.FalseAssertion.json',
        './garak_logs__misleading.txt': None,  # No corresponding JSON file found
        
        # Phrasing attacks
        './garak_logs__phrasing.FutureTense.txt': './http/http_logs_1753115986__phrasing.FutureTense.json',
        
        # Prompt Injection attacks
        './garak_logs__promptinject.HijackHateHumans.txt': './http/http_logs_1753058602__promptinject.HijackHateHumans.json',
        './garak_logs__promptinject.HijackKillHumans.txt': './http/http_logs_1753127064__promptinject.HijackKillHumans.json',
        
        # Snowball attacks
        './garak_logs__snowball.Primes.txt': './http/http_logs_1753045637__snowball.Primes.json',
        
        # Suffix attacks
        './garak_logs__suffix.BEAST.txt': './http/http_logs_1753050667__suffix.BEAST.json',
        
        # Topic attacks
        './garak_logs__topic.WordnetAllowedWords.txt': './http/http_logs_1753051170__topic.WordnetAllowedWords.json',
        
        # XSS attacks
        './garak_logs__xss.ColabAIDataLeakage.txt': './http/http_logs_1753045079__xss.ColabAIDataLeakage.json',
        './garak_logs__xss.StringAssemblyDataExfil.txt': './http/http_logs_1753050800__xss.StringAssemblyDataExfil.json',
    }
    
    print("Processing Garak vulnerability scanner results...")
    print(f"Found {len(file_mapping)} file pairs to process")
    
    successful_processes = 0
    failed_processes = 0
    
    # Process each file pair
    for text_file, json_file in file_mapping.items():
        if process_file_pair(text_file, json_file):
            successful_processes += 1
        else:
            failed_processes += 1
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"BATCH PROCESSING COMPLETE")
    print(f"="*60)
    print(f"Total file pairs: {len(file_mapping)}")
    print(f"Successful processes: {successful_processes}")
    print(f"Failed processes: {failed_processes}")
    
    if failed_processes > 0:
        print(f"\n⚠️  {failed_processes} file pair(s) failed to process.")
        print("Please check the error messages above and ensure all input files exist.")
    else:
        print(f"\n🎉 All file pairs processed successfully!")
        
    # List generated output files
    print(f"\nGenerated output files:")
    for json_file in file_mapping.values():
        output_file = generate_output_filename(json_file)
        print(f"  - {output_file}")
    
    print(f"\nYou can now examine the generated *_failures.json files for detailed results.")


if __name__ == "__main__":
    main()