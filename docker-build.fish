#!/usr/bin/env fish
# Build script for KORA Docker image (Podman/Docker compatible)

# Colors for output
set -l GREEN '\033[0;32m'
set -l BLUE '\033[0;34m'
set -l YELLOW '\033[1;33m'
set -l RED '\033[0;31m'
set -l NC '\033[0m' # No Color

# Detect container runtime
set -l CONTAINER_CMD docker
if command -v podman > /dev/null
    set CONTAINER_CMD podman
    echo -e "$BLUE🐋 Using Podman$NC"
else if command -v docker > /dev/null
    set CONTAINER_CMD docker
    echo -e "$BLUE🐋 Using Docker$NC"
else
    echo -e "$RED❌ Error: Neither podman nor docker found$NC"
    exit 1
end

# Parse arguments
set -l BUILD_ARGS
set -l IMAGE_TAG "kora:latest"
set -l PLATFORM ""

for arg in $argv
    switch $arg
        case '--no-cache'
            set BUILD_ARGS $BUILD_ARGS --no-cache
        case '--platform=*'
            set PLATFORM (string split -m 1 = $arg)[2]
        case '--tag=*' '-t=*'
            set IMAGE_TAG (string split -m 1 = $arg)[2]
        case '--help' '-h'
            echo "Usage: ./docker-build.fish [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-cache              Build without cache"
            echo "  --platform=PLATFORM     Set target platform (e.g., linux/amd64, linux/arm64)"
            echo "  --tag=TAG, -t=TAG       Set image tag (default: kora:latest)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./docker-build.fish"
            echo "  ./docker-build.fish --no-cache"
            echo "  ./docker-build.fish --platform=linux/amd64"
            echo "  ./docker-build.fish --tag=kora:dev"
            exit 0
        case '*'
            echo -e "$RED❌ Unknown option: $arg$NC"
            echo "Use --help for usage information"
            exit 1
    end
end

# Add platform if specified
if test -n "$PLATFORM"
    set BUILD_ARGS $BUILD_ARGS --platform=$PLATFORM
    echo -e "$BLUE🏗️  Building for platform: $PLATFORM$NC"
end

# Start build
echo -e "$GREEN========================================$NC"
echo -e "$GREEN 🚀 Building KORA Docker Image$NC"
echo -e "$GREEN========================================$NC"
echo ""
echo -e "$BLUE📦 Image tag: $IMAGE_TAG$NC"
echo ""

# Build the image
if $CONTAINER_CMD build -t $IMAGE_TAG -f Dockerfile $BUILD_ARGS .
    echo ""
    echo -e "$GREEN========================================$NC"
    echo -e "$GREEN ✅ Build successful!$NC"
    echo -e "$GREEN========================================$NC"
    echo ""
    echo -e "$BLUE📋 Next steps:$NC"
    echo ""
    echo "  1. Run the web interface:"
    echo -e "     $YELLOW$CONTAINER_CMD run -p 7860:7860 -p 7861:7861 -v kora-data:/data $IMAGE_TAG$NC"
    echo ""
    echo "  2. Run the API server:"
    echo -e "     $YELLOW$CONTAINER_CMD run -p 8000:8000 -v kora-data:/data $IMAGE_TAG api$NC"
    echo ""
    echo "  3. Use docker-compose:"
    echo -e "     $YELLOW$CONTAINER_CMD-compose up -d$NC"
    echo ""
    echo "  4. Generate API key:"
    echo -e "     $YELLOW$CONTAINER_CMD run --rm -v kora-data:/data $IMAGE_TAG auth generate --username admin$NC"
    echo ""
else
    echo ""
    echo -e "$RED========================================$NC"
    echo -e "$RED ❌ Build failed!$NC"
    echo -e "$RED========================================$NC"
    exit 1
end
