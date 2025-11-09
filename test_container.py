#!/usr/bin/env python3
"""
KORA Container Testing Script
Tests the KORA container functionality on Podman
"""

import subprocess
import sys
import time
import json
import urllib.request
import urllib.error
from typing import Tuple, Optional

# Configuration
CONTAINER_CMD = "podman"
IMAGE_TAG = "kora:latest"
TEST_VOLUME = "kora-test-data"
WEB_PORT = 17860
ADMIN_PORT = 17861
API_PORT = 18000

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'

# Test statistics
tests_passed = 0
tests_failed = 0
test_results = []


def run_command(cmd: list, capture_output: bool = True, check: bool = False) -> Tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr
    except Exception as e:
        return 1, "", str(e)


def print_section(title: str):
    """Print a test section header"""
    print(f"\n{Colors.BLUE}{'=' * 55}")
    print(f"  Testing: {title}")
    print(f"{'=' * 55}{Colors.NC}")


def test_pass(message: str):
    """Mark a test as passed"""
    global tests_passed
    tests_passed += 1
    result = f"{Colors.GREEN}✅ PASS: {message}{Colors.NC}"
    print(result)
    test_results.append(("PASS", message))


def test_fail(message: str, details: str = ""):
    """Mark a test as failed"""
    global tests_failed
    tests_failed += 1
    result = f"{Colors.RED}❌ FAIL: {message}{Colors.NC}"
    print(result)
    if details:
        print(f"  {details}")
    test_results.append(("FAIL", message))


def cleanup():
    """Clean up test resources"""
    print(f"\n{Colors.YELLOW}🧹 Cleaning up test resources...{Colors.NC}")
    
    # Stop containers
    run_command([CONTAINER_CMD, "stop", "kora-test-web", "kora-test-api"])
    run_command([CONTAINER_CMD, "rm", "kora-test-web", "kora-test-api"])
    
    # Remove volume
    run_command([CONTAINER_CMD, "volume", "rm", TEST_VOLUME])
    
    print(f"{Colors.GREEN}✅ Cleanup complete{Colors.NC}")


def wait_for_url(url: str, timeout: int = 30, interval: int = 2) -> bool:
    """Wait for a URL to become accessible"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(interval)
    return False


def check_url(url: str) -> bool:
    """Check if a URL is accessible"""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except:
        return False


def main():
    print(f"{Colors.GREEN}{'=' * 55}")
    print(f"  KORA Container Test Suite")
    print(f"{'=' * 55}{Colors.NC}")
    print(f"\n{Colors.BLUE}Container runtime: {CONTAINER_CMD}{Colors.NC}\n")

    # Test 1: Check if image exists
    print_section("Image Availability")
    code, stdout, stderr = run_command([CONTAINER_CMD, "images"])
    if "kora" in stdout and "latest" in stdout:
        test_pass("Image 'kora:latest' exists")
    else:
        test_fail("Image 'kora:latest' not found")
        print(f"  {Colors.YELLOW}Run './docker-build.fish' first{Colors.NC}")
        sys.exit(1)

    # Test 2: Create test volume
    print_section("Volume Creation")
    code, stdout, stderr = run_command([CONTAINER_CMD, "volume", "create", TEST_VOLUME])
    if code == 0:
        test_pass("Created test volume")
    else:
        test_fail("Failed to create test volume", stderr)

    # Test 3: Test container startup (web)
    print_section("Web Container Startup")
    cmd = [
        CONTAINER_CMD, "run", "-d",
        "--name", "kora-test-web",
        "-p", f"{WEB_PORT}:7860",
        "-p", f"{ADMIN_PORT}:7861",
        "-v", f"{TEST_VOLUME}:/data",
        "-e", "OLLAMA_HOST=http://host.containers.internal:11434",
        IMAGE_TAG, "web"
    ]
    code, stdout, stderr = run_command(cmd)
    
    if code == 0:
        test_pass("Web container started")
        print(f"  Container ID: {stdout.strip()[:12]}")
    else:
        test_fail("Failed to start web container", stderr)
        cleanup()
        sys.exit(1)

    # Wait for container to initialize
    print(f"{Colors.BLUE}⏳ Waiting for services to start...{Colors.NC}")
    time.sleep(10)

    # Test 4: Check container is running
    print_section("Container Health")
    code, stdout, stderr = run_command([CONTAINER_CMD, "ps"])
    if "kora-test-web" in stdout:
        test_pass("Container is running")
    else:
        test_fail("Container is not running")
        # Show logs
        code, logs, _ = run_command([CONTAINER_CMD, "logs", "kora-test-web"])
        print(f"  Container logs:\n{logs[:500]}")

    # Test 5: Check if web interface is accessible
    print_section("Web Interface Accessibility")
    print(f"  Trying to connect to http://localhost:{WEB_PORT}/")
    
    if wait_for_url(f"http://localhost:{WEB_PORT}/", timeout=30):
        test_pass("Web interface is accessible")
    else:
        test_fail("Web interface is not accessible")
        # Show logs
        code, logs, _ = run_command([CONTAINER_CMD, "logs", "kora-test-web"])
        print(f"  Last 20 lines of logs:")
        print('\n'.join(logs.split('\n')[-20:]))

    # Test 6: Check admin interface
    print_section("Admin Interface")
    if check_url(f"http://localhost:{ADMIN_PORT}/"):
        test_pass("Admin interface is accessible")
    else:
        test_fail("Admin interface is not accessible")

    # Test 7: Test API server
    print_section("API Server Startup")
    cmd = [
        CONTAINER_CMD, "run", "-d",
        "--name", "kora-test-api",
        "-p", f"{API_PORT}:8000",
        "-v", f"{TEST_VOLUME}:/data",
        "-e", "OLLAMA_HOST=http://host.containers.internal:11434",
        IMAGE_TAG, "api"
    ]
    code, stdout, stderr = run_command(cmd)
    
    if code == 0:
        test_pass("API container started")
        print(f"  Container ID: {stdout.strip()[:12]}")
    else:
        test_fail("Failed to start API container", stderr)

    # Wait for API to start
    time.sleep(8)

    # Test 8: Check API accessibility
    print_section("API Accessibility")
    if wait_for_url(f"http://localhost:{API_PORT}/docs", timeout=20):
        test_pass("API documentation is accessible")
    else:
        test_fail("API documentation is not accessible")
        code, logs, _ = run_command([CONTAINER_CMD, "logs", "kora-test-api"])
        print(f"  Last 20 lines of logs:")
        print('\n'.join(logs.split('\n')[-20:]))

    # Test 9: Test auth CLI
    print_section("Auth CLI")
    cmd = [
        CONTAINER_CMD, "run", "--rm",
        "-v", f"{TEST_VOLUME}:/data",
        IMAGE_TAG, "auth", "generate", "--username", "testuser"
    ]
    code, stdout, stderr = run_command(cmd)
    
    if code == 0 and "API Key" in stdout:
        test_pass("Auth CLI works - generated API key")
        # Extract API key for later use
        for line in stdout.split('\n'):
            if "API Key:" in line:
                print(f"  {line}")
    else:
        test_fail("Auth CLI failed", stdout + stderr)

    # Test 10: Test auth list
    print_section("Auth List")
    cmd = [
        CONTAINER_CMD, "run", "--rm",
        "-v", f"{TEST_VOLUME}:/data",
        IMAGE_TAG, "auth", "list"
    ]
    code, stdout, stderr = run_command(cmd)
    
    if code == 0 and "testuser" in stdout:
        test_pass("Auth list works")
    else:
        test_fail("Auth list failed", stdout + stderr)

    # Test 11: Test volume persistence
    print_section("Volume Persistence")
    # Stop and remove web container
    run_command([CONTAINER_CMD, "stop", "kora-test-web"])
    run_command([CONTAINER_CMD, "rm", "kora-test-web"])
    
    # List auth keys again
    cmd = [
        CONTAINER_CMD, "run", "--rm",
        "-v", f"{TEST_VOLUME}:/data",
        IMAGE_TAG, "auth", "list"
    ]
    code, stdout, stderr = run_command(cmd)
    
    if code == 0 and "testuser" in stdout:
        test_pass("Data persists across container restarts")
    else:
        test_fail("Data persistence failed", stdout)

    # Test 12: Check file permissions
    print_section("File Permissions")
    cmd = [
        CONTAINER_CMD, "run", "--rm",
        "-v", f"{TEST_VOLUME}:/data",
        IMAGE_TAG, "bash", "-c", "test -w /data && echo 'writable' || echo 'not writable'"
    ]
    code, stdout, stderr = run_command(cmd)
    
    if "writable" in stdout:
        test_pass("Container has write permissions")
    else:
        test_fail("Container lacks write permissions", stdout)

    # Test 13: Check environment variables
    print_section("Environment Variables")
    cmd = [
        CONTAINER_CMD, "run", "--rm",
        IMAGE_TAG, "bash", "-c", "echo $OLLAMA_HOST"
    ]
    code, stdout, stderr = run_command(cmd)
    
    if "http" in stdout:
        test_pass("Environment variables are set")
        print(f"  OLLAMA_HOST: {stdout.strip()}")
    else:
        test_fail("Environment variables not set properly", stdout)

    # Test 14: Test entrypoint help
    print_section("Entrypoint Script")
    cmd = [CONTAINER_CMD, "run", "--rm", IMAGE_TAG, "invalid-command"]
    code, stdout, stderr = run_command(cmd)
    
    if "Usage:" in stdout or "Usage:" in stderr:
        test_pass("Entrypoint script provides help")
    else:
        test_fail("Entrypoint script doesn't provide help")

    # Test 15: Test container inspect
    print_section("Container Configuration")
    cmd = [CONTAINER_CMD, "inspect", "kora-test-api"]
    code, stdout, stderr = run_command(cmd)
    
    if code == 0:
        try:
            data = json.loads(stdout)
            if data and len(data) > 0:
                test_pass("Container inspection works")
                config = data[0].get("Config", {})
                print(f"  Image: {config.get('Image', 'N/A')}")
                print(f"  User: {config.get('User', 'N/A')}")
        except json.JSONDecodeError:
            test_fail("Failed to parse container inspection data")
    else:
        test_fail("Container inspection failed", stderr)

    # Cleanup
    cleanup()

    # Print summary
    print(f"\n{Colors.BLUE}{'=' * 55}")
    print(f"  Test Summary")
    print(f"{'=' * 55}{Colors.NC}")
    print(f"\n  Total tests:  {tests_passed + tests_failed}")
    print(f"  {Colors.GREEN}✅ Passed:    {tests_passed}{Colors.NC}")
    print(f"  {Colors.RED}❌ Failed:    {tests_failed}{Colors.NC}")
    print()

    # Print detailed results
    if tests_failed > 0:
        print(f"{Colors.RED}Failed tests:{Colors.NC}")
        for status, message in test_results:
            if status == "FAIL":
                print(f"  ❌ {message}")
        print()

    if tests_failed == 0:
        print(f"{Colors.GREEN}{'=' * 55}")
        print(f"  ✅ All tests passed!")
        print(f"{'=' * 55}{Colors.NC}")
        sys.exit(0)
    else:
        print(f"{Colors.RED}{'=' * 55}")
        print(f"  ❌ Some tests failed")
        print(f"{'=' * 55}{Colors.NC}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Tests interrupted by user{Colors.NC}")
        cleanup()
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Unexpected error: {e}{Colors.NC}")
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(1)
