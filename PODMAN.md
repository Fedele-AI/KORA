# KORA Podman Quick Reference

This guide provides Podman-specific commands for running KORA.

## Why Podman?

- **Rootless containers**: Run containers as non-root user
- **No daemon**: No background process required
- **Docker compatibility**: Drop-in replacement for Docker CLI
- **Better security**: SELinux integration, no elevated privileges

## Quick Start with Podman

### Build Image

```fish
# Using the build script (automatically detects Podman)
./docker-build.fish

# Or manually
podman build -t kora:latest -f Dockerfile .
```

### Run Container

```fish
# Using the run script (automatically detects Podman)
./docker-run.fish

# Or manually - Web interface
podman run --rm -p 7860:7860 -p 7861:7861 \
  -v kora-data:/data \
  -e OLLAMA_HOST=http://host.containers.internal:11434 \
  kora:latest

# API server
podman run --rm -p 8000:8000 \
  -v kora-data:/data \
  -e OLLAMA_HOST=http://host.containers.internal:11434 \
  kora:latest api
```

### Podman Compose

```fish
# Install podman-compose if not already installed
pip install podman-compose

# Or use system package manager
brew install podman-compose  # macOS
sudo apt install podman-compose  # Ubuntu/Debian

# Run with podman-compose
podman-compose up -d
podman-compose logs -f kora-web
podman-compose down
```

## GPU Support with Podman

### NVIDIA GPU

```fish
# Install NVIDIA Container Toolkit
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Run with GPU using the script
./docker-run.fish --gpu

# Or manually
podman run --rm \
  --device nvidia.com/gpu=all \
  -p 7860:7860 -p 7861:7861 \
  -v kora-data:/data \
  kora:latest
```

### AMD GPU (ROCm)

```fish
# Install ROCm support
# See: https://rocmdocs.amd.com/

# Run with AMD GPU
podman run --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -p 7860:7860 -p 7861:7861 \
  -v kora-data:/data \
  kora:latest
```

## Podman-Specific Features

### Rootless Containers

```fish
# Run as non-root (default in Podman)
podman run --rm -p 7860:7860 -v kora-data:/data kora:latest

# Check if running rootless
podman info | grep rootless
```

### Pod Management

Create a pod for all KORA services:

```fish
# Create a pod
podman pod create --name kora-pod -p 7860:7860 -p 7861:7861 -p 8000:8000

# Run web interface in the pod
podman run -d --pod kora-pod --name kora-web \
  -v kora-data:/data \
  kora:latest web

# Run API in the same pod
podman run -d --pod kora-pod --name kora-api \
  -v kora-data:/data \
  kora:latest api

# Manage the entire pod
podman pod stop kora-pod
podman pod start kora-pod
podman pod rm kora-pod
```

### Systemd Integration

Generate systemd unit files:

```fish
# Generate systemd service for container
podman generate systemd --new --files --name kora-web

# Move to systemd directory
mkdir -p ~/.config/systemd/user
mv container-kora-web.service ~/.config/systemd/user/

# Enable and start
systemctl --user daemon-reload
systemctl --user enable --now container-kora-web

# Check status
systemctl --user status container-kora-web
```

### Auto-update Containers

```fish
# Label image for auto-update
podman build -t kora:latest --label "io.containers.autoupdate=registry" .

# Run with auto-update label
podman run -d --name kora-web \
  --label "io.containers.autoupdate=registry" \
  -p 7860:7860 -p 7861:7861 \
  -v kora-data:/data \
  kora:latest

# Enable auto-update timer
systemctl --user enable --now podman-auto-update.timer

# Manual update
podman auto-update
```

## Volume Management

```fish
# Create volume
podman volume create kora-data

# Inspect volume
podman volume inspect kora-data

# List volumes
podman volume ls

# Backup volume
podman run --rm -v kora-data:/data -v (pwd):/backup \
  alpine tar czf /backup/kora-backup.tar.gz /data

# Restore volume
podman run --rm -v kora-data:/data -v (pwd):/backup \
  alpine tar xzf /backup/kora-backup.tar.gz -C /

# Remove volume
podman volume rm kora-data
```

## Networking

### Host Network (Linux only)

```fish
# Use host network for better performance
podman run --rm --network host \
  -v kora-data:/data \
  -e OLLAMA_HOST=http://localhost:11434 \
  kora:latest
```

### Custom Network

```fish
# Create custom network
podman network create kora-network

# Run with custom network
podman run -d --name kora-web \
  --network kora-network \
  -p 7860:7860 -p 7861:7861 \
  -v kora-data:/data \
  kora:latest

# Connect Ollama to same network (if containerized)
podman run -d --name ollama \
  --network kora-network \
  -v ollama-data:/root/.ollama \
  ollama/ollama

# Update OLLAMA_HOST to use network
podman run --rm \
  --network kora-network \
  -p 7860:7860 -p 7861:7861 \
  -v kora-data:/data \
  -e OLLAMA_HOST=http://ollama:11434 \
  kora:latest
```

## Security Features

### SELinux Labels

```fish
# Use SELinux labels for better security
podman run --rm \
  -p 7860:7860 -p 7861:7861 \
  -v kora-data:/data:Z \
  kora:latest

# :Z - Private unshared label (recommended)
# :z - Shared label
```

### User Namespace

```fish
# Verify user namespace mapping
podman unshare cat /proc/self/uid_map

# Run with specific UID/GID mapping
podman run --rm \
  --uidmap 0:100000:65536 \
  --gidmap 0:100000:65536 \
  -p 7860:7860 -v kora-data:/data \
  kora:latest
```

### Read-only Root Filesystem

```fish
# Run with read-only root filesystem
podman run --rm --read-only \
  -p 7860:7860 -p 7861:7861 \
  -v kora-data:/data \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  kora:latest
```

## Troubleshooting

### Check Podman Version

```fish
podman --version
podman info
```

### Connection to Host Services

On Linux, use `host.containers.internal`:
```fish
-e OLLAMA_HOST=http://host.containers.internal:11434
```

Alternatively, find host IP:
```fish
# Get default route IP
ip route | grep default | awk '{print $3}'

# Use that IP
-e OLLAMA_HOST=http://172.17.0.1:11434
```

### Port Binding Issues (Rootless)

If ports < 1024 don't work:

```fish
# Allow binding to privileged ports
sudo sysctl net.ipv4.ip_unprivileged_port_start=80

# Or use higher ports and redirect
iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8080
```

### Storage Driver

```fish
# Check storage driver
podman info | grep graphDriverName

# Change to overlay (if needed)
mkdir -p ~/.config/containers
echo '[storage]
driver = "overlay"' > ~/.config/containers/storage.conf
```

## Migration from Docker

Podman is mostly compatible with Docker commands:

```fish
# Create alias (add to ~/.config/fish/config.fish)
alias docker='podman'

# Or use podman-docker package
sudo apt install podman-docker  # Ubuntu/Debian
brew install podman-docker      # macOS
```

## Resources

- [Podman Documentation](https://docs.podman.io/)
- [Podman vs Docker](https://docs.podman.io/en/latest/markdown/podman.1.html)
- [Rootless Containers](https://rootlesscontaine.rs/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

## Examples

### Complete Development Setup

```fish
# Build image
./docker-build.fish

# Create volume
podman volume create kora-data

# Run web interface in background
podman run -d --name kora-web \
  -p 7860:7860 -p 7861:7861 \
  -v kora-data:/data \
  -v (pwd)/RAG:/app/RAG:ro \
  -e OLLAMA_HOST=http://host.containers.internal:11434 \
  kora:latest

# Generate API key
podman run --rm \
  -v kora-data:/data \
  kora:latest auth generate --username admin

# View logs
podman logs -f kora-web

# Access at http://localhost:7860
```

### Production Setup with Systemd

```fish
# Build image
./docker-build.fish

# Generate systemd unit
podman run -d --name kora-web \
  -p 7860:7860 -p 7861:7861 \
  -v kora-data:/data \
  kora:latest

podman generate systemd --new --files --name kora-web

# Install service
mkdir -p ~/.config/systemd/user
mv container-kora-web.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now container-kora-web

# Enable lingering (keep services running after logout)
loginctl enable-linger $USER
```
