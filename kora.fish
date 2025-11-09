#!/usr/bin/env fish
# Makefile alternative for Fish shell - Common KORA operations

# Colors
set -l GREEN '\033[0;32m'
set -l BLUE '\033[0;34m'
set -l YELLOW '\033[1;33m'
set -l RED '\033[0;31m'
set -l NC '\033[0m'

# Detect container runtime
set -g CONTAINER_CMD docker
if command -v podman > /dev/null
    set CONTAINER_CMD podman
else if command -v docker > /dev/null
    set CONTAINER_CMD docker
else
    echo -e "$RED❌ Error: Neither podman nor docker found$NC"
    exit 1
end

# Configuration
set -g IMAGE_TAG "kora:latest"
set -g DATA_VOLUME "kora-data"

function show_help
    echo -e "$BLUE════════════════════════════════════════════════$NC"
    echo -e "$BLUE KORA - Container Management Script$NC"
    echo -e "$BLUE════════════════════════════════════════════════$NC"
    echo ""
    echo "Usage: ./kora.fish <command> [options]"
    echo ""
    echo "Commands:"
    echo ""
    echo "  🔨 Build Commands:"
    echo "    build              Build the Docker image"
    echo "    build-no-cache     Build without cache"
    echo ""
    echo "  🚀 Run Commands:"
    echo "    run                Run web interface (interactive)"
    echo "    start              Start web interface (background)"
    echo "    api                Run API server (interactive)"
    echo "    start-api          Start API server (background)"
    echo ""
    echo "  🔑 Auth Commands:"
    echo "    auth-generate      Generate API key"
    echo "    auth-list          List all API keys"
    echo ""
    echo "  📊 Management Commands:"
    echo "    logs               View web interface logs"
    echo "    logs-api           View API server logs"
    echo "    stop               Stop all services"
    echo "    restart            Restart all services"
    echo "    shell              Open shell in container"
    echo ""
    echo "  📦 Docker Compose Commands:"
    echo "    up                 Start all services with compose"
    echo "    down               Stop all services with compose"
    echo "    ps                 Show running containers"
    echo ""
    echo "  🧹 Cleanup Commands:"
    echo "    clean              Stop and remove containers"
    echo "    clean-all          Clean containers and volumes"
    echo "    prune              Remove unused images"
    echo ""
    echo "  ℹ️  Info Commands:"
    echo "    status             Show service status"
    echo "    volumes            List volumes"
    echo "    images             List images"
    echo ""
    echo "Examples:"
    echo "  ./kora.fish build           # Build the image"
    echo "  ./kora.fish start           # Start in background"
    echo "  ./kora.fish logs            # View logs"
    echo "  ./kora.fish auth-generate   # Generate API key"
    echo ""
end

function build
    echo -e "$GREEN🔨 Building KORA image...$NC"
    ./docker-build.fish $argv
end

function build_no_cache
    echo -e "$GREEN🔨 Building KORA image (no cache)...$NC"
    ./docker-build.fish --no-cache
end

function run_web
    echo -e "$GREEN🚀 Starting web interface (interactive)...$NC"
    ./docker-run.fish --service=web $argv
end

function start_web
    echo -e "$GREEN🚀 Starting web interface (background)...$NC"
    ./docker-run.fish --service=web --detach $argv
    echo ""
    echo -e "$BLUE🌐 Access at:$NC"
    echo -e "  Main UI:    ${YELLOW}http://localhost:7860$NC"
    echo -e "  Admin UI:   ${YELLOW}http://localhost:7861$NC"
end

function run_api
    echo -e "$GREEN🔌 Starting API server (interactive)...$NC"
    ./docker-run.fish --service=api $argv
end

function start_api
    echo -e "$GREEN🔌 Starting API server (background)...$NC"
    ./docker-run.fish --service=api --detach $argv
    echo ""
    echo -e "$BLUE🌐 Access at:$NC"
    echo -e "  API:        ${YELLOW}http://localhost:8000$NC"
    echo -e "  Docs:       ${YELLOW}http://localhost:8000/docs$NC"
end

function auth_generate
    echo -e "$GREEN🔑 Generating API key...$NC"
    echo ""
    echo -n "Enter username: "
    read -l username
    
    if test -z "$username"
        set username "user"
    end
    
    $CONTAINER_CMD run --rm -v $DATA_VOLUME:/data $IMAGE_TAG auth generate --username $username
end

function auth_list
    echo -e "$GREEN🔑 Listing API keys...$NC"
    $CONTAINER_CMD run --rm -v $DATA_VOLUME:/data $IMAGE_TAG auth list
end

function show_logs
    echo -e "$BLUE📋 Showing web interface logs...$NC"
    $CONTAINER_CMD logs -f kora-web
end

function show_logs_api
    echo -e "$BLUE📋 Showing API server logs...$NC"
    $CONTAINER_CMD logs -f kora-api
end

function stop_services
    echo -e "$YELLOW🛑 Stopping services...$NC"
    $CONTAINER_CMD stop kora-web kora-api 2>/dev/null
    echo -e "$GREEN✅ Services stopped$NC"
end

function restart_services
    echo -e "$YELLOW🔄 Restarting services...$NC"
    stop_services
    sleep 2
    start_web
end

function open_shell
    echo -e "$BLUE🐚 Opening shell in new container...$NC"
    $CONTAINER_CMD run --rm -it -v $DATA_VOLUME:/data $IMAGE_TAG bash
end

function compose_up
    echo -e "$GREEN🚀 Starting services with compose...$NC"
    $CONTAINER_CMD-compose up -d
    echo ""
    echo -e "$GREEN✅ Services started$NC"
    echo ""
    echo -e "$BLUE🌐 Access at:$NC"
    echo -e "  Main UI:    ${YELLOW}http://localhost:7860$NC"
    echo -e "  Admin UI:   ${YELLOW}http://localhost:7861$NC"
    echo -e "  API:        ${YELLOW}http://localhost:8000$NC"
end

function compose_down
    echo -e "$YELLOW🛑 Stopping compose services...$NC"
    $CONTAINER_CMD-compose down
    echo -e "$GREEN✅ Services stopped$NC"
end

function show_ps
    echo -e "$BLUE📊 Running containers:$NC"
    $CONTAINER_CMD ps --filter "name=kora"
end

function clean_containers
    echo -e "$YELLOW🧹 Cleaning containers...$NC"
    $CONTAINER_CMD stop kora-web kora-api 2>/dev/null
    $CONTAINER_CMD rm kora-web kora-api 2>/dev/null
    echo -e "$GREEN✅ Containers cleaned$NC"
end

function clean_all
    echo -e "$RED🧹 Cleaning containers and volumes...$NC"
    echo -e "$YELLOW⚠️  This will delete all KORA data!$NC"
    echo -n "Are you sure? (yes/no): "
    read -l confirm
    
    if test "$confirm" = "yes"
        clean_containers
        $CONTAINER_CMD volume rm $DATA_VOLUME 2>/dev/null
        echo -e "$GREEN✅ Cleanup complete$NC"
    else
        echo -e "$BLUEℹ️  Cleanup cancelled$NC"
    end
end

function prune_images
    echo -e "$YELLOW🧹 Removing unused images...$NC"
    $CONTAINER_CMD image prune -f
    echo -e "$GREEN✅ Unused images removed$NC"
end

function show_status
    echo -e "$BLUE📊 KORA Status:$NC"
    echo ""
    echo "Containers:"
    $CONTAINER_CMD ps -a --filter "name=kora" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "Volumes:"
    $CONTAINER_CMD volume ls --filter "name=kora"
    echo ""
    echo "Images:"
    $CONTAINER_CMD images --filter "reference=kora"
end

function show_volumes
    echo -e "$BLUE💾 KORA Volumes:$NC"
    $CONTAINER_CMD volume ls --filter "name=kora"
    echo ""
    echo "Volume details:"
    $CONTAINER_CMD volume inspect $DATA_VOLUME 2>/dev/null || echo "No volumes found"
end

function show_images
    echo -e "$BLUE📦 KORA Images:$NC"
    $CONTAINER_CMD images --filter "reference=kora"
end

# Main command dispatcher
if test (count $argv) -eq 0
    show_help
    exit 0
end

set -l command $argv[1]
set -e argv[1]

switch $command
    case build
        build $argv
    case build-no-cache
        build_no_cache
    case run
        run_web $argv
    case start
        start_web $argv
    case api
        run_api $argv
    case start-api
        start_api $argv
    case auth-generate
        auth_generate
    case auth-list
        auth_list
    case logs
        show_logs
    case logs-api
        show_logs_api
    case stop
        stop_services
    case restart
        restart_services
    case shell
        open_shell
    case up
        compose_up
    case down
        compose_down
    case ps
        show_ps
    case clean
        clean_containers
    case clean-all
        clean_all
    case prune
        prune_images
    case status
        show_status
    case volumes
        show_volumes
    case images
        show_images
    case help -h --help
        show_help
    case '*'
        echo -e "$RED❌ Unknown command: $command$NC"
        echo ""
        show_help
        exit 1
end
