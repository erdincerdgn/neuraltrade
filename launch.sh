#!/bin/bash
# ============================================================
# NeuralTrade - Launch Scripts
# ============================================================
# Tüm build ve deploy komutları
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="neuraltrade"
IMAGE_TAG="latest"
REGISTRY=""  # Docker Hub, GCR, ECR etc.

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
echo_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================
# 1. DEBUG: Run all tests
# ============================================================
run_tests() {
    echo_info "🧪 Running debug tests..."
    python tests/debug_all.py
    echo_success "Tests completed!"
}

# ============================================================
# 2. DOCKER: Build image
# ============================================================
docker_build() {
    echo_info "🐳 Building Docker image..."
    docker build -t ${PROJECT_NAME}:${IMAGE_TAG} .
    echo_success "Docker image built: ${PROJECT_NAME}:${IMAGE_TAG}"
}

# ============================================================
# 3. DOCKER: Push to registry
# ============================================================
docker_push() {
    if [ -z "$REGISTRY" ]; then
        echo_warn "No registry set. Skipping push."
        return
    fi
    
    echo_info "📤 Pushing to registry..."
    docker tag ${PROJECT_NAME}:${IMAGE_TAG} ${REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}
    docker push ${REGISTRY}/${PROJECT_NAME}:${IMAGE_TAG}
    echo_success "Image pushed to ${REGISTRY}"
}

# ============================================================
# 4. DOCKER-COMPOSE: Start all services
# ============================================================
compose_up() {
    echo_info "🚀 Starting Docker Compose stack..."
    docker-compose up -d
    echo_success "Stack started!"
    echo_info "Services:"
    echo "  - Bot: http://localhost:8000"
    echo "  - Metrics: http://localhost:9090"
    echo "  - Redis: localhost:6379"
}

# ============================================================
# 5. DOCKER-COMPOSE: Start with monitoring
# ============================================================
compose_up_full() {
    echo_info "🚀 Starting full stack with monitoring..."
    docker-compose --profile monitoring up -d
    echo_success "Full stack started!"
    echo_info "Services:"
    echo "  - Bot: http://localhost:8000"
    echo "  - Prometheus: http://localhost:9091"
    echo "  - Grafana: http://localhost:3000 (admin/neuraltrade123)"
}

# ============================================================
# 6. DOCKER-COMPOSE: Run tests
# ============================================================
compose_test() {
    echo_info "🧪 Running tests in Docker..."
    docker-compose --profile test run --rm neuraltrade-test
    echo_success "Tests completed!"
}

# ============================================================
# 7. DOCKER-COMPOSE: Run pipeline
# ============================================================
compose_pipeline() {
    echo_info "🔄 Running pipeline in Docker..."
    docker-compose --profile pipeline run --rm neuraltrade-pipeline
}

# ============================================================
# 8. DOCKER-COMPOSE: Stop all
# ============================================================
compose_down() {
    echo_info "🛑 Stopping all services..."
    docker-compose --profile monitoring down
    echo_success "All services stopped."
}

# ============================================================
# 9. KUBERNETES: Deploy
# ============================================================
k8s_deploy() {
    echo_info "☸️ Deploying to Kubernetes..."
    
    # Apply all manifests
    kubectl apply -f k8s/neuraltrade-deployment.yaml
    
    echo_success "Deployed to Kubernetes!"
    echo_info "Check status:"
    echo "  kubectl get pods -n neuraltrade"
    echo "  kubectl get svc -n neuraltrade"
}

# ============================================================
# 10. KUBERNETES: Status
# ============================================================
k8s_status() {
    echo_info "☸️ Kubernetes status..."
    kubectl get all -n neuraltrade
}

# ============================================================
# 11. KUBERNETES: Logs
# ============================================================
k8s_logs() {
    echo_info "📜 Bot logs..."
    kubectl logs -f -n neuraltrade deployment/neuraltrade-bot
}

# ============================================================
# 12. KUBERNETES: Delete
# ============================================================
k8s_delete() {
    echo_warn "⚠️ Deleting Kubernetes deployment..."
    kubectl delete -f k8s/neuraltrade-deployment.yaml
    echo_success "Deleted."
}

# ============================================================
# 13. KUBERNETES: Port forward
# ============================================================
k8s_port_forward() {
    echo_info "🔗 Port forwarding..."
    kubectl port-forward -n neuraltrade svc/neuraltrade-svc 8000:8000 &
    kubectl port-forward -n neuraltrade svc/neuraltrade-svc 9090:9090 &
    echo_success "Forwarded:"
    echo "  - API: http://localhost:8000"
    echo "  - Metrics: http://localhost:9090"
}

# ============================================================
# HELP
# ============================================================
show_help() {
    echo ""
    echo "============================================================"
    echo "🚀 NeuralTrade Launch Commands"
    echo "============================================================"
    echo ""
    echo "LOCAL:"
    echo "  test              Run debug tests"
    echo ""
    echo "DOCKER:"
    echo "  docker-build      Build Docker image"
    echo "  docker-push       Push to registry"
    echo ""
    echo "DOCKER-COMPOSE:"
    echo "  up                Start basic stack (bot + redis)"
    echo "  up-full           Start with monitoring (+ prometheus, grafana)"
    echo "  test              Run tests in Docker"
    echo "  pipeline          Run pipeline in Docker"
    echo "  down              Stop all services"
    echo ""
    echo "KUBERNETES:"
    echo "  k8s-deploy        Deploy to Kubernetes"
    echo "  k8s-status        Show status"
    echo "  k8s-logs          Show bot logs"
    echo "  k8s-delete        Delete deployment"
    echo "  k8s-forward       Port forward to local"
    echo ""
}

# ============================================================
# MAIN
# ============================================================
case "$1" in
    "test")
        run_tests
        ;;
    "docker-build")
        docker_build
        ;;
    "docker-push")
        docker_push
        ;;
    "up")
        compose_up
        ;;
    "up-full")
        compose_up_full
        ;;
    "compose-test")
        compose_test
        ;;
    "pipeline")
        compose_pipeline
        ;;
    "down")
        compose_down
        ;;
    "k8s-deploy")
        k8s_deploy
        ;;
    "k8s-status")
        k8s_status
        ;;
    "k8s-logs")
        k8s_logs
        ;;
    "k8s-delete")
        k8s_delete
        ;;
    "k8s-forward")
        k8s_port_forward
        ;;
    *)
        show_help
        ;;
esac
