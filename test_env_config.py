#!/usr/bin/env python3
"""
Test script to validate environment variable configuration system
"""

import os
import tempfile
import subprocess
import sys

def test_environment_variables():
    """Test that environment variables properly configure the system"""
    print("🧪 Testing environment variable configuration...")
    
    # Test script content
    test_script = '''
import os
import sys
sys.path.insert(0, ".")

# Set custom environment variables
os.environ["CREWGRAPH_SYSTEM_USER"] = "production_user"
os.environ["CREWGRAPH_ORGANIZATION"] = "ACME Corp"
os.environ["CREWGRAPH_ENVIRONMENT"] = "staging"

from crewgraph_ai.config import get_current_user
from crewgraph_ai.memory.dict_memory import DictMemory

# Test that config picks up environment variables
user = get_current_user()
assert user == "production_user", f"Expected 'production_user', got '{user}'"
print(f"✅ Environment variable config: {user}")

# Test memory with custom user
memory = DictMemory()
memory.connect()
health = memory.get_health()
assert health["checked_by"] == "production_user", f"Expected 'production_user', got '{health['checked_by']}'"
print(f"✅ Memory using environment user: {health['checked_by']}")
memory.disconnect()

print("✅ Environment variable configuration working!")
'''
    
    # Write test script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        # Run the test script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd='.')
        
        print("Test output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Environment variable test passed!")
            return True
        else:
            print("❌ Environment variable test failed!")
            return False
            
    finally:
        # Clean up
        os.unlink(script_path)

def main():
    """Run environment variable tests"""
    print("🚀 Testing CrewGraph AI Environment Variable Configuration")
    print("=" * 60)
    
    try:
        success = test_environment_variables()
        
        if success:
            print("\n" + "=" * 60)
            print("✅ ENVIRONMENT CONFIGURATION TEST PASSED!")
            print("🎉 The system properly responds to environment variables.")
            print("\n📋 Supported Environment Variables:")
            print("  • CREWGRAPH_SYSTEM_USER - Override default system user")
            print("  • CREWGRAPH_ORGANIZATION - Set organization name")
            print("  • CREWGRAPH_ENVIRONMENT - Set environment (dev/staging/production)")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)