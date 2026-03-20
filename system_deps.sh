#!/usr/bin/env bash
set -euo pipefail

INSTALL_PYTHON=true
INSTALL_GO=true
INSTALL_NODE=true
INSTALL_DOTNET=true
INSTALL_CPP=true
INSTALL_RUST=true

GO_VERSION="1.22.5"
NODE_MAJOR="20"

say() {
  printf "%s\n" "$1"
}

say "🧰 Installing system dependencies for rl-coding-agent"
sudo apt-get update
sudo apt-get install -y curl ca-certificates gnupg lsb-release software-properties-common apt-transport-https

if [ "$INSTALL_PYTHON" = true ]; then
  say "🐍 Installing Python 3.11 and pip"
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-venv python3-pip
fi

if [ "$INSTALL_GO" = true ]; then
  say "🐹 Installing Go ${GO_VERSION}"
  curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tar.gz
  sudo rm -rf /usr/local/go
  sudo tar -C /usr/local -xzf /tmp/go.tar.gz
  grep -q '/usr/local/go/bin' "$HOME/.bashrc" || printf '\nexport PATH="/usr/local/go/bin:$PATH"\n' >> "$HOME/.bashrc"
fi

if [ "$INSTALL_NODE" = true ]; then
  say "🟩 Installing Node.js ${NODE_MAJOR} LTS"
  curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | sudo -E bash -
  sudo apt-get install -y nodejs
fi

if [ "$INSTALL_DOTNET" = true ]; then
  say "🔵 Installing .NET 8 SDK"
  curl -fsSL https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb -o /tmp/packages-microsoft-prod.deb
  sudo dpkg -i /tmp/packages-microsoft-prod.deb
  sudo apt-get update
  sudo apt-get install -y dotnet-sdk-8.0
fi

if [ "$INSTALL_CPP" = true ]; then
  say "⚙️ Installing build-essential and g++"
  sudo apt-get install -y build-essential g++
fi

if [ "$INSTALL_RUST" = true ]; then
  say "🦀 Installing Rust stable via rustup"
  curl https://sh.rustup.rs -sSf | sh -s -- -y
  grep -q 'cargo/env' "$HOME/.bashrc" || printf '\nsource "$HOME/.cargo/env"\n' >> "$HOME/.bashrc"
fi

say "✅ All selected runtimes are installed."
say "👉 Run: source ~/.bashrc"
say "👉 Then: pip install -r requirements.txt"
