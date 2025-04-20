# 1. Create a Kind cluster if you don't have one
kind create cluster --name ai-chatbot

# 2. Build the Docker image
docker build -t ai-chatbot:latest .

# 3. Load the image into Kind cluster
kind load docker-image ai-chatbot:latest --name ai-chatbot

# 4. Update the secret in k8s-deployment.yaml with your actual Gemini API key
# Edit the file and replace "your-gemini-api-key" with your actual key

# 5. Apply the Kubernetes manifests
kubectl apply -f k8s-deployment.yaml

# 6. Check if the pod is running
kubectl get pods

# 7. Expose the service to your local machine
kubectl port-forward service/ai-chatbot 8080:80

# The app will now be accessible at http://localhost:8080

# Additional Commands, use it if required.

# View logs of the pod
kubectl logs -l app=ai-chatbot

# Get detailed information about the pod
kubectl describe pod -l app=ai-chatbot

# Scale the deployment
kubectl scale deployment ai-chatbot --replicas=2

# Delete the deployment when done
kubectl delete -f k8s-deployment.yaml

# Delete the cluster when completely done
kind delete cluster --name ai-chatbot