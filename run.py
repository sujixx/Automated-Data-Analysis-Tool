#!/usr/bin/env python3
"""
AI Analytics Platform - Quick Start Script
==========================================

This script helps you quickly set up and run your AI Analytics Platform.

Usage:
    python run.py

What it does:
1. Checks if all required packages are installed
2. Installs missing packages
3. Launches the Streamlit app
4. Opens your browser to the app

Author: AI Analytics Platform
Version: 1.0
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version}")
        return True

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_requirements():
    """Check and install required packages"""
    required_packages = [
        "streamlit",
        "pandas",
        "numpy", 
        "matplotlib",
        "seaborn",
        "plotly",
        "scikit-learn",
        "openpyxl",
        "xlsxwriter"
    ]
    
    print("🔍 Checking required packages...")
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def create_sample_data():
    """Create sample data file if it doesn't exist"""
    sample_file = "sample_data.csv"
    
    if not os.path.exists(sample_file):
        print("📊 Creating sample data file...")
        
        sample_data = """Name,Age,Salary,Department,Years_Experience,City,Performance_Score
Alice Johnson,25,50000,IT,3,New York,85
Bob Smith,30,60000,HR,5,Los Angeles,78
Charlie Brown,35,70000,IT,8,Chicago,92
Diana Prince,28,55000,Finance,4,Houston,88
Eve Wilson,32,65000,HR,6,Phoenix,82
Frank Miller,29,58000,IT,4,Philadelphia,79
Grace Lee,31,62000,Marketing,5,San Antonio,86
Henry Davis,27,52000,Finance,3,San Diego,81
Ivy Chen,33,68000,IT,7,Dallas,90
Jack Taylor,26,51000,HR,2,San Jose,75
Karen White,34,72000,Marketing,8,Austin,94
Liam Garcia,28,56000,Finance,4,Jacksonville,83
Mia Rodriguez,30,61000,IT,5,Fort Worth,87
Noah Martinez,29,59000,HR,4,Columbus,80
Olivia Anderson,31,63000,Marketing,6,Charlotte,89"""
        
        with open(sample_file, 'w') as f:
            f.write(sample_data)
        
        print(f"✅ Sample data created: {sample_file}")
    else:
        print(f"✅ Sample data already exists: {sample_file}")

def check_required_files():
    """Check if main app file exists"""
    required_files = ["app.py"]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please make sure you have:")
        print("- app.py (main Streamlit application)")
        print("- nlp_to_sql.py (optional, for advanced NLP features)")
        print("- advanced_viz.py (optional, for advanced visualizations)")
        return False
    
    return True

def run_streamlit_app():
    """Launch the Streamlit app"""
    print("\n🚀 Launching AI Analytics Platform...")
    print("📱 The app will open in your default web browser")
    print("🌐 URL: http://localhost:8501")
    print("\n💡 Tips:")
    print("- Use Ctrl+C to stop the server")
    print("- Refresh the browser page if something doesn't work")
    print("- Check the terminal for any error messages")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Shutting down the server...")
        return True
    
    return True

def main():
    """Main function to run the setup and launch process"""
    print("🚀 AI Analytics Platform - Quick Start")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check required files
    if not check_required_files():
        return
    
    # Install requirements
    if not check_and_install_requirements():
        print("❌ Failed to install required packages")
        return
    
    # Create sample data
    create_sample_data()
    
    print("\n✅ Setup completed successfully!")
    print("\n" + "="*50)
    
    # Ask user if they want to run the app
    response = input("🚀 Do you want to launch the app now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '']:
        run_streamlit_app()
    else:
        print("\n💻 To run the app later, use: streamlit run app.py")
        print("📚 Check the tutorial section in the app for help!")

if __name__ == "__main__":
    main()