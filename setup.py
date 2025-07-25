#!/usr/bin/env python3
"""
Setup script for LoRA DataGen
Initializes the project with necessary directories and files
"""

import os
from pathlib import Path
import shutil
from colorama import Fore, init

# Initialize colorama
init()

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        "pdfs",        # Input PDFs
        "data",        # Generated datasets
        "outputs",     # Final outputs
        "logs"         # Log files
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"{Fore.GREEN}✅ Created directory: {directory}{Fore.RESET}")
            
            # Create .gitkeep files to maintain directory structure
            gitkeep = path / ".gitkeep"
            gitkeep.touch()
        else:
            print(f"{Fore.YELLOW}📁 Directory already exists: {directory}{Fore.RESET}")

def setup_config():
    """Setup configuration file"""
    config_example = Path("config.yaml.example")
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        if config_example.exists():
            shutil.copy(config_example, config_file)
            print(f"{Fore.GREEN}✅ Created config.yaml from example{Fore.RESET}")
            print(f"{Fore.YELLOW}⚠️  Remember to edit config.yaml with your settings{Fore.RESET}")
        else:
            print(f"{Fore.RED}❌ config.yaml.example not found{Fore.RESET}")
    else:
        print(f"{Fore.YELLOW}⚙️ config.yaml already exists{Fore.RESET}")

def check_environment():
    """Check environment setup"""
    print(f"\n{Fore.CYAN}🔍 Checking Environment{Fore.RESET}")
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"{Fore.GREEN}✅ OPENAI_API_KEY is set{Fore.RESET}")
    else:
        print(f"{Fore.RED}❌ OPENAI_API_KEY not found in environment{Fore.RESET}")
        print(f"{Fore.YELLOW}💡 Get your API key from: https://platform.openai.com/api-keys{Fore.RESET}")
        print(f"{Fore.YELLOW}💡 Set it with: export OPENAI_API_KEY='your-key-here'{Fore.RESET}")
    
    # Check Python version
    import sys
    if sys.version_info >= (3, 8):
        print(f"{Fore.GREEN}✅ Python {sys.version.split()[0]} (compatible){Fore.RESET}")
    else:
        print(f"{Fore.RED}❌ Python {sys.version.split()[0]} (requires 3.8+){Fore.RESET}")

def create_example_files():
    """Create example files for testing"""
    
    # Create a simple example instruction file for quality testing
    example_instruction = {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
    }
    
    instruction_file = Path("data/instruction.json")
    instruction_file.parent.mkdir(exist_ok=True)
    
    if not instruction_file.exists():
        import json
        with open(instruction_file, 'w', encoding='utf-8') as f:
            json.dump([example_instruction], f, indent=2)
        print(f"{Fore.GREEN}✅ Created example instruction.json{Fore.RESET}")

def main():
    """Main setup function"""
    print(f"{Fore.CYAN}🚀 LoRA DataGen Setup{Fore.RESET}")
    print(f"{Fore.CYAN}📚 Initializing project structure...{Fore.RESET}\n")
    
    # Create directories
    create_directory_structure()
    
    # Setup configuration
    print(f"\n{Fore.CYAN}⚙️ Setting up configuration{Fore.RESET}")
    setup_config()
    
    # Create example files
    print(f"\n{Fore.CYAN}📝 Creating example files{Fore.RESET}")
    create_example_files()
    
    # Check environment
    check_environment()
    
    # Final instructions
    print(f"\n{Fore.GREEN}🎉 Setup Complete!{Fore.RESET}")
    print(f"\n{Fore.CYAN}📋 Next Steps:{Fore.RESET}")
    print(f"1. {Fore.YELLOW}Edit config.yaml with your settings{Fore.RESET}")
    print(f"2. {Fore.YELLOW}Place PDF files in ./pdfs/ directory{Fore.RESET}")
    print(f"3. {Fore.YELLOW}Set OPENAI_API_KEY environment variable{Fore.RESET}")
    print(f"4. {Fore.YELLOW}Run: python syntheticdatageneration_openai.py{Fore.RESET}")
    print(f"\n{Fore.MAGENTA}📖 See README.md for detailed instructions{Fore.RESET}")

if __name__ == "__main__":
    main() 