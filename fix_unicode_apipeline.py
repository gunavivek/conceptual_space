#!/usr/bin/env python3
"""
Fix Unicode issues in A-Pipeline scripts for Windows compatibility
"""

import os
import glob

def fix_unicode_in_file(filepath):
    """Replace Unicode characters with ASCII alternatives"""
    
    replacements = {
        '✓': '[OK]',
        '✗': '[FAIL]',
        '→': '->',
        '📊': '[STATS]',
        '🎯': '[TARGET]',
        '⚠️': '[WARNING]',
        '🔄': '[REFRESH]',
        '📈': '[CHART]',
        '🎉': '[SUCCESS]'
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        for unicode_char, ascii_replacement in replacements.items():
            content = content.replace(unicode_char, ascii_replacement)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {os.path.basename(filepath)}")
            return True
        else:
            print(f"No changes needed: {os.path.basename(filepath)}")
            return False
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    # Get all Python files in A-Pipeline scripts
    script_dir = "A_Concept_pipeline/scripts"
    pattern = os.path.join(script_dir, "*.py")
    
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} Python files in A-Pipeline scripts")
    print("-" * 60)
    
    fixed_count = 0
    for filepath in files:
        if fix_unicode_in_file(filepath):
            fixed_count += 1
    
    print("-" * 60)
    print(f"Fixed {fixed_count} files")
    print("Unicode issues resolved for Windows compatibility!")

if __name__ == "__main__":
    main()