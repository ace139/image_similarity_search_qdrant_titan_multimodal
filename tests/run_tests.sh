#!/bin/bash

# AWS Access Tests Runner
# Convenient script to run individual tests or all tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

print_header() {
    echo -e "\n${BLUE}${BOLD}$1${NC}"
    echo -e "${BLUE}$(printf '=%.0s' {1..50})${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

show_usage() {
    cat << EOF
${BOLD}AWS Access Tests Runner${NC}

Usage: $0 [OPTION]

Options:
    quick       Run quick health check only (~5s)
    cli         Run comprehensive CLI tests (~30s)
    boto3       Run boto3/Python SDK tests (~15s)
    simulation  Run full app workflow simulation (~45s)
    all         Run all tests in sequence (~95s)
    help        Show this help message

Examples:
    $0 quick          # Fast health check
    $0 simulation     # Full end-to-end test
    $0 all           # Run everything

EOF
}

run_quick_test() {
    print_header "🚀 Quick AWS Health Check"
    cd "$ROOT_DIR"
    
    if [ -x "tests/quick_aws_tests.sh" ]; then
        ./tests/quick_aws_tests.sh
        print_success "Quick test completed"
    else
        print_error "quick_aws_tests.sh not found or not executable"
        return 1
    fi
}

run_cli_test() {
    print_header "🔧 Comprehensive CLI Tests"
    cd "$ROOT_DIR"
    
    if [ -x "tests/test_aws_access.sh" ]; then
        ./tests/test_aws_access.sh
        print_success "CLI tests completed"
    else
        print_error "test_aws_access.sh not found or not executable"
        return 1
    fi
}

run_boto3_test() {
    print_header "🐍 Boto3/Python SDK Tests"
    cd "$ROOT_DIR"
    
    if [ -f "tests/test_boto3_access.py" ]; then
        python tests/test_boto3_access.py
        print_success "Boto3 tests completed"
    else
        print_error "test_boto3_access.py not found"
        return 1
    fi
}

run_simulation_test() {
    print_header "🖼️ Full Application Workflow Simulation"
    cd "$ROOT_DIR"
    
    if [ -f "tests/test_app_simulation.py" ]; then
        python tests/test_app_simulation.py
        print_success "Simulation tests completed"
    else
        print_error "test_app_simulation.py not found"
        return 1
    fi
}

run_all_tests() {
    print_header "🎯 Running All AWS Access Tests"
    print_info "This will take approximately 95 seconds..."
    
    echo -e "\n${BOLD}Test Sequence:${NC}"
    echo "1. Quick Health Check (~5s)"
    echo "2. CLI Verification (~30s)"  
    echo "3. Boto3 SDK Tests (~15s)"
    echo "4. Full Workflow Simulation (~45s)"
    echo ""
    
    run_quick_test
    echo ""
    
    run_cli_test
    echo ""
    
    run_boto3_test
    echo ""
    
    run_simulation_test
    
    print_header "🎉 All Tests Completed Successfully!"
    echo -e "${GREEN}${BOLD}Your AWS setup is fully verified and ready for production!${NC}"
}

check_prerequisites() {
    cd "$ROOT_DIR"
    
    # Check .env file
    if [ ! -f ".env" ]; then
        print_error ".env file not found in project root"
        echo "Please create a .env file with your AWS configuration"
        return 1
    fi
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_error "Python not found"
        echo "Please install Python to run boto3 tests"
        return 1
    fi
    
    # Check if AWS CLI is available
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI not found"
        echo "Please install AWS CLI to run CLI tests"
        return 1
    fi
    
    return 0
}

main() {
    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
    # Parse command line argument
    case "${1:-help}" in
        "quick")
            run_quick_test
            ;;
        "cli")
            run_cli_test
            ;;
        "boto3")
            run_boto3_test
            ;;
        "simulation")
            run_simulation_test
            ;;
        "all")
            run_all_tests
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Make test files executable if they aren't already
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

# Run main function
main "$@"
