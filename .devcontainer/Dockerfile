FROM ubuntu:latest

# Update package lists
RUN apt-get update

# Install necessary packages
RUN apt-get install -y curl apt-transport-https ca-certificates gnupg

# # Install Docker
# RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
# RUN apt-get update
# RUN apt-get install -y docker-ce docker-ce-cli containerd.io

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
RUN install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Kind
RUN curl -Lo ./kind "https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64"
RUN install -o root -g root -m 0755 kind /usr/local/bin/kind

# Install Helm (optional)
RUN curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
RUN chmod 700 get_helm.sh
RUN ./get_helm.sh

# Clean up apt packages
RUN apt-get autoremove -y && apt-get clean -y

# USER vscode

# Set the working directory
# WORKDIR /workspaces

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
