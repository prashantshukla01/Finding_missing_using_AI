# check_cctv_content.py
import os

cctv_file = "models/cctv_manager.py"
if os.path.exists(cctv_file):
    print(f"=== Content of {cctv_file} ===")
    with open(cctv_file, 'r') as f:
        content = f.read()
        # Show first 500 characters to see the class definition
        print(content[:1000])
        print("\n=== Methods in file ===")
        # Count methods
        methods = [line for line in content.split('\n') if 'def ' in line and '#' not in line]
        for method in methods[:10]:  # Show first 10 methods
            print(method.strip())
        print(f"Total methods found: {len(methods)}")
else:
    print(f"File {cctv_file} does not exist!")