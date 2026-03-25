FROM python:3.12-slim

# Install git and other system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
# We would copy uv.lock if we had one, but we can also just run uv sync
# Initialize git in the vault directory preemptively just to avoid issues
RUN mkdir -p marvin_vault && cd marvin_vault && git init

# Install dependencies into system environment so uv run isn't strictly necessary 
# if we wanted, but uv run handles the virtualenv perfectly.
COPY . .
RUN uv sync

# Default command for the gateway
CMD ["uv", "run", "marvin", "--transport", "sse", "--host", "0.0.0.0", "--port", "8421"]
