# Kubernetes CLI (kubectl) Commands and Examples

This document provides commonly used kubectl commands with examples and corresponding YAML definitions.

1. Basic Commands
kubectl get: Retrieves information about Kubernetes resources.

```bash

Get all pods in the default namespace
kubectl get pods

Get all deployments in the 'production' namespace
kubectl get deployments -n production

Get details about a specific pod
kubectl get pod my-pod -o yaml

Get all resources of a particular type (e.g., all services) in all namespaces
kubectl get services --all-namespaces
```

kubectl describe: Provides detailed information about a Kubernetes resource. Useful for troubleshooting.

```bash

Describe a pod to see its events, labels, and container status
kubectl describe pod my-pod

Describe a service to see its endpoints and selector
kubectl describe service my-service
```

kubectl create: Creates Kubernetes resources from YAML or JSON files.

```bash

Create a resource from a YAML file
kubectl create -f my-deployment.yaml

Create multiple resources from a directory of YAML files
kubectl create -f my-resources/
```

kubectl apply: Updates Kubernetes resources from YAML or JSON files. Creates a resource if it doesn't exist, updates it if it does. Recommended for managing resources declaratively.

```bash

Apply a YAML file to update a resource or create it if it doesn't exist
kubectl apply -f my-deployment.yaml

Apply a directory of YAML files
kubectl apply -f my-resources/
```

kubectl delete: Deletes Kubernetes resources.

```bash

Delete a pod
kubectl delete pod my-pod

Delete a deployment
kubectl delete deployment my-deployment

Delete a resource using a YAML file
kubectl delete -f my-service.yaml

Delete all pods with a specific label
kubectl delete pods -l app=my-app

Force delete a pod that is stuck in terminating state
kubectl delete pod my-pod --force --grace-period=0
```

kubectl exec: Executes a command inside a container in a pod.

```bash

Execute a shell inside a container
kubectl exec -it my-pod -- /bin/bash

Execute a command inside a container
kubectl exec my-pod -- ls /app

Execute a command in a specific container within the pod
kubectl exec -it my-pod -c my-container -- /bin/bash
```

kubectl logs: Retrieves logs from a container in a pod.

```bash

Get logs from a pod's container
kubectl logs my-pod

Follow the logs in real-time
kubectl logs -f my-pod

Get logs from a specific container within the pod
kubectl logs my-pod -c my-container

Get logs from the last 10 minutes
kubectl logs --since=10m my-pod
```

kubectl expose: Creates a new service to expose a deployment, service, replication controller, or pod.

```bash

Expose a deployment as a service
kubectl expose deployment my-deployment --type=LoadBalancer --port=80 --target-port=8080

Expose a pod as a service
kubectl expose pod my-pod --port=80 --target-port=8080 --name=my-service
```

kubectl scale: Scales the number of replicas for a deployment, replication controller, or replica set.

```bash

Scale a deployment to 5 replicas
kubectl scale deployment my-deployment --replicas=5

Scale a replica set to 3 replicas
kubectl scale replicaset my-replicaset --replicas=3
```

kubectl rollout: Manages the rollout of a deployment.

```bash

Check the rollout status of a deployment
kubectl rollout status deployment my-deployment

Undo the last rollout
kubectl rollout undo deployment my-deployment

Pause a rollout
kubectl rollout pause deployment my-deployment

Resume a rollout
kubectl rollout resume deployment my-deployment

Restart a rollout
kubectl rollout restart deployment my-deployment
```

kubectl port-forward: Forwards a local port to a port on a pod. Useful for accessing services running inside the cluster from your local machine.

```bash

Forward local port 8080 to port 80 on the my-pod pod
kubectl port-forward my-pod 8080:80
```

2. YAML Examples
2.1. Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-container
        image: nginx:latest
        ports:
        - containerPort: 80
2.2. Service
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
  type: LoadBalancer # or ClusterIP, NodePort
2.3. Pod
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
  labels:
    app: my-app
spec:
  containers:
  - name: my-container
    image: nginx:latest
    ports:
    - containerPort: 80
2.4. ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-configmap
data:
  my_variable: "my_value"
  another_variable: "another_value"
2.5. Secret
apiVersion: v1
kind: Secret
metadata:
  name: my-secret
type: Opaque
data:
  my_username: $(echo -n 'my_username' | base64)
  my_password: $(echo -n 'my_password' | base64)
Important notes on secrets:

Store sensitive information like passwords, API keys, and certificates.
Data is stored encoded in base64, not encrypted. Therefore, rely on access control mechanisms to restrict access to secrets.
Never commit secrets to source control.
3. Namespaces
kubectl get namespaces: List all namespaces.

bash kubectl get namespaces

kubectl create namespace: Create a new namespace.

bash kubectl create namespace my-namespace

kubectl delete namespace: Delete a namespace.

bash kubectl delete namespace my-namespace

--namespace or -n: Specify the namespace for a command.

bash kubectl get pods -n my-namespace

4. Common Options
Option	Description	Example
-n, --namespace	Specifies the namespace for the operation.	kubectl get pods -n my-namespace
-f, --filename	Specifies the filename containing the resource definition.	kubectl create -f my-deployment.yaml
-o, --output	Specifies the output format (e.g., yaml, json, wide).	kubectl get pods -o yaml
-l, --selector	Specifies a label selector to filter resources.	kubectl get pods -l app=my-app
--all-namespaces	Applies the operation to all namespaces.	kubectl get pods --all-namespaces
-i, --interactive	Uses interactive terminal for interactive inputs, or --stdin=true	kubectl exec -it my-pod -- bash
-t	TTY, must be combined with -i	kubectl exec -it my-pod -- bash
5. Advanced Examples
Updating images:

bash kubectl set image deployment/my-deployment my-container=nginx:1.21
This command updates the image of the my-container container in the my-deployment deployment to nginx:1.21.

Applying changes without restarting pods:

bash kubectl apply -f my-deployment.yaml --record

The --record flag keeps track of changes and allows you to rollback. --record is deprecated and should be replaced by rollout history functionality.

Rolling back a deployment:

bash kubectl rollout undo deployment/my-deployment

Using JSONPath for complex output:

bash kubectl get pod my-pod -o jsonpath='{.status.podIP}'
This command retrieves only the IP address of the pod.

Connecting to a database pod:

bash kubectl exec -it my-db-pod -- mysql -u root -p

This document provides a starting point. Kubernetes is a complex system, and continued learning is key. Refer to the official Kubernetes documentation for the most up-to-date and detailed information. kubectl help is also very useful!