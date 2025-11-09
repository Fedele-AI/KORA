#!/usr/bin/env fish
# Test script for KORA Docker setup

# Colors
set -l GREEN '\033[0;32m'
set -l BLUE '\033[0;34m'
set -l YELLOW '\033[1;33m'
set -l RED '\033[0;31m'
set -l NC '\033[0m'

# Detect container runtime
set -l CONTAINER_CMD docker
if command -v podman > /dev/null
    set CONTAINER_CMD podman
else if command -v docker > /dev/null
    set CONTAINER_CMD docker
else
    echo -e "$RED❌ Error: Neither podman nor docker found$NC"
    exit 1
end

set -l IMAGE_TAG "kora:latest"
set -l TEST_VOLUME "kora-test-data"
set -l FAILED_TESTS 0
set -l PASSED_TESTS 0

function print_test_header
    echo ""
    echo -e "$BLUE═══════════════════════════════════════════════════$NC"
    echo -e "$BLUE  Testing: $argv[1]$NC"
    echo -e "$BLUE═══════════════════════════════════════════════════$NC"
end

function test_pass
    echo -e "$GREEN✅ PASS: $argv[1]$NC"
    set PASSED_TESTS (math $PASSED_TESTS + 1)
end

function test_fail
    echo -e "$RED❌ FAIL: $argv[1]$NC"
    set FAILED_TESTS (math $FAILED_TESTS + 1)
end

function cleanup_test
    echo -e "$YELLOW🧹 Cleaning up test resources...$NC"
    $CONTAINER_CMD stop kora-test-web kora-test-api 2>/dev/null
    $CONTAINER_CMD rm kora-test-web kora-test-api 2>/dev/null
    $CONTAINER_CMD volume rm $TEST_VOLUME 2>/dev/null
    echo -e "$GREEN✅ Cleanup complete$NC"
end

# Trap to cleanup on exit
trap cleanup_test EXIT

echo -e "$GREEN═══════════════════════════════════════════════════$NC"
echo -e "$GREEN  KORA Docker Test Suite$NC"
echo -e "$GREEN═══════════════════════════════════════════════════$NC"
echo ""
echo -e "$BLUE Container runtime: $CONTAINER_CMD$NC"
echo ""

# Test 1: Check if image exists
print_test_header "Image Availability"
if $CONTAINER_CMD images | grep -q "kora.*latest"
    test_pass "Image 'kora:latest' exists"
else
    test_fail "Image 'kora:latest' not found"
    echo -e "$YELLOW  Run './docker-build.fish' first$NC"
    exit 1
end

# Test 2: Create test volume
print_test_header "Volume Creation"
if $CONTAINER_CMD volume create $TEST_VOLUME > /dev/null 2>&1
    test_pass "Created test volume"
else
    test_fail "Failed to create test volume"
end

# Test 3: Test container startup
print_test_header "Container Startup"
set -l container_id ($CONTAINER_CMD run -d --name kora-test-web \
    -p 17860:7860 -p 17861:7861 \
    -v $TEST_VOLUME:/data \
    -e OLLAMA_HOST=http://host.docker.internal:11434 \
    $IMAGE_TAG web 2>&1)

if test $status -eq 0
    test_pass "Web container started"
    sleep 5  # Give it time to start
else
    test_fail "Failed to start web container"
    echo "  Error: $container_id"
end

# Test 4: Check container is running
print_test_header "Container Health"
if $CONTAINER_CMD ps | grep -q kora-test-web
    test_pass "Container is running"
else
    test_fail "Container is not running"
    echo "  Logs:"
    $CONTAINER_CMD logs kora-test-web
end

# Test 5: Check if web interface is accessible
print_test_header "Web Interface Accessibility"
sleep 5  # Give more time for Gradio to start
if curl -sf http://localhost:17860/ > /dev/null 2>&1
    test_pass "Web interface is accessible"
else
    test_fail "Web interface is not accessible"
    echo "  Trying to connect to http://localhost:17860/"
    echo "  Container logs:"
    $CONTAINER_CMD logs kora-test-web | tail -20
end

# Test 6: Check admin interface
print_test_header "Admin Interface"
if curl -sf http://localhost:17861/ > /dev/null 2>&1
    test_pass "Admin interface is accessible"
else
    test_fail "Admin interface is not accessible"
end

# Test 7: Test API server
print_test_header "API Server"
set -l api_container_id ($CONTAINER_CMD run -d --name kora-test-api \
    -p 18000:8000 \
    -v $TEST_VOLUME:/data \
    -e OLLAMA_HOST=http://host.docker.internal:11434 \
    $IMAGE_TAG api 2>&1)

if test $status -eq 0
    test_pass "API container started"
    sleep 5
else
    test_fail "Failed to start API container"
end

# Test 8: Check API accessibility
print_test_header "API Accessibility"
sleep 3
if curl -sf http://localhost:18000/docs > /dev/null 2>&1
    test_pass "API documentation is accessible"
else
    test_fail "API documentation is not accessible"
    echo "  Container logs:"
    $CONTAINER_CMD logs kora-test-api | tail -20
end

# Test 9: Test auth CLI
print_test_header "Auth CLI"
set -l auth_output ($CONTAINER_CMD run --rm \
    -v $TEST_VOLUME:/data \
    $IMAGE_TAG auth generate --username testuser 2>&1)

if echo $auth_output | grep -q "API Key"
    test_pass "Auth CLI works - generated API key"
else
    test_fail "Auth CLI failed"
    echo "  Output: $auth_output"
end

# Test 10: Test volume persistence
print_test_header "Volume Persistence"
$CONTAINER_CMD stop kora-test-web > /dev/null 2>&1
$CONTAINER_CMD rm kora-test-web > /dev/null 2>&1

set -l list_output ($CONTAINER_CMD run --rm \
    -v $TEST_VOLUME:/data \
    $IMAGE_TAG auth list 2>&1)

if echo $list_output | grep -q "testuser"
    test_pass "Data persists across container restarts"
else
    test_fail "Data persistence failed"
    echo "  Output: $list_output"
end

# Test 11: Check file permissions
print_test_header "File Permissions"
set -l perm_check ($CONTAINER_CMD run --rm \
    -v $TEST_VOLUME:/data \
    $IMAGE_TAG bash -c "test -w /data && echo 'writable' || echo 'not writable'" 2>&1)

if echo $perm_check | grep -q "writable"
    test_pass "Container has write permissions"
else
    test_fail "Container lacks write permissions"
end

# Test 12: Check environment variables
print_test_header "Environment Variables"
set -l env_check ($CONTAINER_CMD run --rm \
    $IMAGE_TAG bash -c "echo \$OLLAMA_HOST" 2>&1)

if echo $env_check | grep -q "http"
    test_pass "Environment variables are set"
else
    test_fail "Environment variables not set properly"
end

# Test 13: Test entrypoint script
print_test_header "Entrypoint Script"
set -l help_output ($CONTAINER_CMD run --rm $IMAGE_TAG invalid-command 2>&1)

if echo $help_output | grep -q "Usage:"
    test_pass "Entrypoint script provides help"
else
    test_fail "Entrypoint script doesn't provide help"
end

# Test 14: Test RAG mount
print_test_header "RAG Directory Mount"
if test -d ./RAG
    set -l rag_test ($CONTAINER_CMD run --rm \
        -v (pwd)/RAG:/app/RAG:ro \
        $IMAGE_TAG bash -c "test -d /app/RAG && echo 'exists' || echo 'missing'" 2>&1)
    
    if echo $rag_test | grep -q "exists"
        test_pass "RAG directory mounts correctly"
    else
        test_fail "RAG directory mount failed"
    end
else
    echo -e "$YELLOW⚠️  SKIP: RAG directory not found$NC"
end

# Summary
echo ""
echo -e "$BLUE═══════════════════════════════════════════════════$NC"
echo -e "$BLUE  Test Summary$NC"
echo -e "$BLUE═══════════════════════════════════════════════════$NC"
echo ""
echo -e "  Total tests:  $(math $PASSED_TESTS + $FAILED_TESTS)"
echo -e "  $GREEN✅ Passed:    $PASSED_TESTS$NC"
echo -e "  $RED❌ Failed:    $FAILED_TESTS$NC"
echo ""

if test $FAILED_TESTS -eq 0
    echo -e "$GREEN═══════════════════════════════════════════════════$NC"
    echo -e "$GREEN  ✅ All tests passed!$NC"
    echo -e "$GREEN═══════════════════════════════════════════════════$NC"
    exit 0
else
    echo -e "$RED═══════════════════════════════════════════════════$NC"
    echo -e "$RED  ❌ Some tests failed$NC"
    echo -e "$RED═══════════════════════════════════════════════════$NC"
    exit 1
end
