#!/usr/bin/env fish
# Run script for KORA Docker container (Podman/Docker compatible)

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

# Default values
set -l SERVICE "web"
set -l IMAGE_TAG "kora:latest"
set -l CONTAINER_NAME "kora"
set -l DATA_VOLUME "kora-data"
set -l RAG_MOUNT ""
set -l GPU_ARGS ""
set -l DETACH ""
set -l EXTRA_ARGS

# Check if RAG directory exists
if test -d ./RAG
    set RAG_MOUNT "-v" (pwd)/RAG:/app/RAG:ro
    echo -e "$BLUE📚 RAG directory found, will mount it$NC"
end

# Parse arguments
set -l i 1
while test $i -le (count $argv)
    set -l arg $argv[$i]
    
    switch $arg
        case '--service=*'
            set SERVICE (string split -m 1 = $arg)[2]
        case '--tag=*' '-t=*'
            set IMAGE_TAG (string split -m 1 = $arg)[2]
        case '--name=*'
            set CONTAINER_NAME (string split -m 1 = $arg)[2]
        case '--volume=*' '-v=*'
            set DATA_VOLUME (string split -m 1 = $arg)[2]
        case '--gpu'
            if test $CONTAINER_CMD = "docker"
                set GPU_ARGS --gpus all
            else if test $CONTAINER_CMD = "podman"
                set GPU_ARGS --device nvidia.com/gpu=all
            end
            echo -e "$YELLOW🎮 GPU support enabled (requires NVIDIA Container Toolkit)$NC"
        case '-d' '--detach'
            set DETACH "-d"
        case '--help' '-h'
            echo "Usage: ./docker-run.fish [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --service=SERVICE       Service to run: web, api, admin, auth (default: web)"
            echo "  --tag=TAG, -t=TAG       Image tag to use (default: kora:latest)"
            echo "  --name=NAME             Container name (default: kora)"
            echo "  --volume=NAME, -v=NAME  Data volume name (default: kora-data)"
            echo "  --gpu                   Enable GPU support (requires NVIDIA Container Toolkit)"
            echo "  -d, --detach            Run container in background"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Services:"
            echo "  web     - Web interface on ports 7860 (main) and 7861 (admin)"
            echo "  api     - REST API on port 8000"
            echo "  admin   - Admin panel only on port 7861"
            echo "  auth    - Run auth CLI commands (pass additional args after --)"
            echo ""
            echo "Examples:"
            echo "  ./docker-run.fish                           # Run web interface"
            echo "  ./docker-run.fish --service=api             # Run API server"
            echo "  ./docker-run.fish --gpu -d                  # Run with GPU in background"
            echo "  ./docker-run.fish --service=auth -- generate --username admin"
            exit 0
        case '--'
            # Everything after -- goes to the container
            set i (math $i + 1)
            set EXTRA_ARGS $argv[$i..-1]
            break
        case '*'
            echo -e "$RED❌ Unknown option: $arg$NC"
            echo "Use --help for usage information"
            exit 1
    end
    
    set i (math $i + 1)
end

# Configure ports based on service
set -l PORT_ARGS
switch $SERVICE
    case 'web'
        set PORT_ARGS -p 7860:7860 -p 7861:7861
        set CONTAINER_NAME "$CONTAINER_NAME-web"
    case 'api'
        set PORT_ARGS -p 8000:8000
        set CONTAINER_NAME "$CONTAINER_NAME-api"
    case 'admin'
        set PORT_ARGS -p 7861:7861
        set CONTAINER_NAME "$CONTAINER_NAME-admin"
    case 'auth'
        set PORT_ARGS ""
        set CONTAINER_NAME "$CONTAINER_NAME-auth"
        set DETACH "" # Never detach for auth commands
end

# Start container
echo -e "$GREEN========================================$NC"
echo -e "$GREEN 🚀 Starting KORA Container$NC"
echo -e "$GREEN========================================$NC"
echo ""
echo -e "$BLUE📦 Image: $IMAGE_TAG$NC"
echo -e "$BLUE🏷️  Service: $SERVICE$NC"
echo -e "$BLUE💾 Data volume: $DATA_VOLUME$NC"
echo ""

# Build the command
set -l CMD $CONTAINER_CMD run --rm $DETACH \
    --name $CONTAINER_NAME \
    $PORT_ARGS \
    -v $DATA_VOLUME:/data \
    $RAG_MOUNT \
    $GPU_ARGS \
    -e OLLAMA_HOST=http://host.docker.internal:11434 \
    $IMAGE_TAG \
    $SERVICE \
    $EXTRA_ARGS

# Run the container
if eval $CMD
    echo ""
    if test -n "$DETACH"
        echo -e "$GREEN✅ Container started in background$NC"
        echo ""
        echo -e "$BLUE📋 Management commands:$NC"
        echo -e "  View logs:  $YELLOW$CONTAINER_CMD logs -f $CONTAINER_NAME$NC"
        echo -e "  Stop:       $YELLOW$CONTAINER_CMD stop $CONTAINER_NAME$NC"
        echo -e "  Remove:     $YELLOW$CONTAINER_CMD rm $CONTAINER_NAME$NC"
    else
        echo -e "$GREEN✅ Container exited$NC"
    end
    
    if test "$SERVICE" = "web"
        echo ""
        echo -e "$BLUE🌐 Access the application at:$NC"
        echo -e "  Main UI:    ${YELLOW}http://localhost:7860$NC"
        echo -e "  Admin UI:   ${YELLOW}http://localhost:7861$NC"
    else if test "$SERVICE" = "api"
        echo ""
        echo -e "$BLUE🌐 Access the API at:$NC"
        echo -e "  API:        ${YELLOW}http://localhost:8000$NC"
        echo -e "  Docs:       ${YELLOW}http://localhost:8000/docs$NC"
    end
    echo ""
else
    echo ""
    echo -e "$RED❌ Failed to start container$NC"
    exit 1
end
