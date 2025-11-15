#!/usr/bin/env python3
"""
Comprehensive test runner for VerifAI project.
Runs all pytest tests in the tests folder.
"""
import sys
import os
import argparse
from pathlib import Path
from typing import List

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")

def get_project_root() -> Path:
    """Get the project root directory"""
    script_dir = Path(__file__).parent.absolute()
    # If script is in scripts/, go up one level
    if script_dir.name == "scripts":
        return script_dir.parent
    # Otherwise assume we're in project root
    return script_dir

def check_pytest_available() -> bool:
    """Check if pytest is available"""
    try:
        import pytest
        return True
    except ImportError:
        return False

def discover_tests() -> List[str]:
    """
    Discover all pytest test files in the tests directory.
    
    Returns:
        List of test file paths
    """
    project_root = get_project_root()
    tests_dir = project_root / "tests"
    
    if not tests_dir.exists():
        return []
    
    test_files = []
    for test_file in tests_dir.glob("test_*.py"):
        # Skip __pycache__ and other non-test files
        if test_file.name.startswith("test_"):
            test_files.append(str(test_file))
    
    return sorted(test_files)

def run_pytest_tests(
    test_files: List[str] = None,
    verbose: bool = True,
    stop_on_failure: bool = False,
    markers: List[str] = None,
    exclude_markers: List[str] = None
) -> int:
    """
    Run pytest tests.
    
    Args:
        test_files: List of specific test files to run (None = all tests)
        verbose: Verbose output
        stop_on_failure: Stop on first failure
        markers: Only run tests with these markers (e.g., ["slow"])
        exclude_markers: Skip tests with these markers (e.g., ["slow"])
    
    Returns:
        Exit code (0 = success, non-zero = failure)
    """
    if not check_pytest_available():
        print_error("pytest not available. Please install it: pip install pytest")
        return 1
    
    project_root = get_project_root()
    tests_dir = project_root / "tests"
    
    # Change to project root for correct imports
    os.chdir(str(project_root))
    
    pytest_args = []
    
    if verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-q")
    
    if stop_on_failure:
        pytest_args.append("-x")
    
    # Add markers
    if markers:
        for marker in markers:
            pytest_args.extend(["-m", marker])
    
    if exclude_markers:
        for marker in exclude_markers:
            pytest_args.extend(["-m", f"not {marker}"])
    
    # Add test files or directory
    if test_files:
        # Filter to only existing test files
        existing_tests = []
        for test_file in test_files:
            test_path = tests_dir / test_file if not Path(test_file).is_absolute() else Path(test_file)
            if test_path.exists():
                existing_tests.append(str(test_path))
            else:
                print_warning(f"Test file not found: {test_file}")
        
        if not existing_tests:
            print_warning("No test files found to run")
            return 1
        
        pytest_args.extend(existing_tests)
        print_info(f"Running specific tests: {', '.join([Path(t).name for t in existing_tests])}")
    else:
        # Run all tests in tests directory
        pytest_args.append(str(tests_dir))
        print_info("Running all tests in tests directory")
    
    try:
        import pytest
        exit_code = pytest.main(pytest_args)
        return exit_code
    except Exception as e:
        print_error(f"Error running pytest: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description="Run all pytest tests for VerifAI project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_all_tests.py              # Run all tests
  python scripts/run_all_tests.py --fast        # Skip slow tests
  python scripts/run_all_tests.py --slow        # Run only slow tests
  python scripts/run_all_tests.py test_basic.py # Run specific test file
  python scripts/run_all_tests.py -x            # Stop on first failure
  python scripts/run_all_tests.py -q            # Quiet output
        """
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Run only slow tests (marked with @pytest.mark.slow)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow tests (exclude tests marked with @pytest.mark.slow)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet output"
    )
    parser.add_argument(
        "-x", "--stop-on-failure",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        help="Run specific test files (e.g., test_integration.py test_basic.py)"
    )
    
    args = parser.parse_args()
    
    # Handle verbose/quiet
    verbose = not args.quiet if args.quiet else args.verbose
    
    # Determine markers
    markers = None
    exclude_markers = None
    
    if args.slow:
        markers = ["slow"]
    elif args.fast:
        exclude_markers = ["slow"]
    
    print_header("VerifAI Test Suite Runner")
    
    # Discover all tests if not specified
    if args.tests:
        test_files = args.tests
    else:
        discovered = discover_tests()
        test_files = [Path(t).name for t in discovered]
    
    if not test_files:
        print_warning("No tests found!")
        return 1
    
    # Run tests
    exit_code = run_pytest_tests(
        test_files=test_files if args.tests else None,
        verbose=verbose,
        stop_on_failure=args.stop_on_failure,
        markers=markers,
        exclude_markers=exclude_markers
    )
    
    # Print summary
    print_header("Test Summary")
    
    if exit_code == 0:
        print_success("All tests passed!")
    else:
        print_error("Some tests failed")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
