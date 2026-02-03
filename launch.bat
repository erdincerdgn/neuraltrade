@echo off
REM ============================================================
REM NeuralTrade - Windows Launch Commands
REM ============================================================

setlocal enabledelayedexpansion

set PROJECT_NAME=neuraltrade
set IMAGE_TAG=latest

echo.
echo ============================================================
echo 🚀 NEURALTRADE LAUNCH COMMANDS
echo ============================================================
echo.

if "%1"=="" goto :help
if "%1"=="test" goto :test
if "%1"=="docker-build" goto :docker_build
if "%1"=="up" goto :compose_up
if "%1"=="up-full" goto :compose_up_full
if "%1"=="down" goto :compose_down
if "%1"=="k8s-deploy" goto :k8s_deploy
if "%1"=="k8s-status" goto :k8s_status
if "%1"=="k8s-logs" goto :k8s_logs
if "%1"=="k8s-delete" goto :k8s_delete
goto :help

:test
echo [INFO] 🧪 Running debug tests...
python tests\debug_all.py
echo [SUCCESS] Tests completed!
goto :end

:docker_build
echo [INFO] 🐳 Building Docker image...
docker build -t %PROJECT_NAME%:%IMAGE_TAG% .
echo [SUCCESS] Docker image built: %PROJECT_NAME%:%IMAGE_TAG%
goto :end

:compose_up
echo [INFO] 🚀 Starting Docker Compose stack...
docker-compose up -d
echo [SUCCESS] Stack started!
echo.
echo Services:
echo   - Bot: http://localhost:8000
echo   - Metrics: http://localhost:9090
echo   - Redis: localhost:6379
goto :end

:compose_up_full
echo [INFO] 🚀 Starting full stack with monitoring...
docker-compose --profile monitoring up -d
echo [SUCCESS] Full stack started!
echo.
echo Services:
echo   - Bot: http://localhost:8000
echo   - Prometheus: http://localhost:9091
echo   - Grafana: http://localhost:3000 (admin/neuraltrade123)
goto :end

:compose_down
echo [INFO] 🛑 Stopping all services...
docker-compose --profile monitoring down
echo [SUCCESS] All services stopped.
goto :end

:k8s_deploy
echo [INFO] ☸️ Deploying to Kubernetes...
kubectl apply -f k8s\neuraltrade-deployment.yaml
echo [SUCCESS] Deployed to Kubernetes!
echo.
echo Check status:
echo   kubectl get pods -n neuraltrade
goto :end

:k8s_status
echo [INFO] ☸️ Kubernetes status...
kubectl get all -n neuraltrade
goto :end

:k8s_logs
echo [INFO] 📜 Bot logs...
kubectl logs -f -n neuraltrade deployment/neuraltrade-bot
goto :end

:k8s_delete
echo [WARN] ⚠️ Deleting Kubernetes deployment...
kubectl delete -f k8s\neuraltrade-deployment.yaml
echo [SUCCESS] Deleted.
goto :end

:help
echo.
echo ============================================================
echo USAGE: launch.bat [command]
echo ============================================================
echo.
echo LOCAL:
echo   test              Run debug tests
echo.
echo DOCKER:
echo   docker-build      Build Docker image
echo.
echo DOCKER-COMPOSE:
echo   up                Start basic stack (bot + redis)
echo   up-full           Start with monitoring
echo   down              Stop all services
echo.
echo KUBERNETES:
echo   k8s-deploy        Deploy to Kubernetes
echo   k8s-status        Show status
echo   k8s-logs          Show bot logs
echo   k8s-delete        Delete deployment
echo.
echo ============================================================
echo.
echo QUICK START:
echo   1. launch.bat test          (Run tests)
echo   2. launch.bat docker-build  (Build image)
echo   3. launch.bat up            (Start with Docker)
echo   4. launch.bat k8s-deploy    (Deploy to K8s)
echo ============================================================

:end
