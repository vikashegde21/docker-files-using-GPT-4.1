# Docker Container Setup Instructions

1. **Clone or copy the files** into a directory (e.g., `flask-redis-api`).
2. **Ensure Docker and Docker Compose are installed** on your system.
3. **Directory structure:**
   ```
   flask-redis-api/
     |- app.py
     |- requirements.txt
     |- Dockerfile
     |- docker-compose.yml
   ```
4. **Build and start the services:**
   ```sh
   docker-compose up --build
   ```
5. **Test the API:**
   - Set a value:
     ```sh
     curl -X POST -H "Content-Type: application/json" \
       -d '{"key": "foo", "value": "bar"}' \
       http://localhost:5000/set
     ```
   - Get a value:
     ```sh
     curl http://localhost:5000/get/foo
     ```
   - Health check:
     ```sh
     curl http://localhost:5000/health
     ```
6. **Stop the services:**
   ```sh
   docker-compose down
   ```

**Note:** Redis data is persisted in a Docker volume (`redis_data`).


## Explanation

### Analysis & Choices
- **Base Image:** Uses `python:3.11-slim` for a minimal, secure Python environment.
- **Multi-stage Build:** Installs build dependencies only in the builder stage, keeping the final image small and secure.
- **Non-root User:** Runs the Flask app as a non-root user (`appuser`) for security.
- **Environment Variables:** Allows configuration of Redis connection via environment variables.
- **Healthcheck:** Both Dockerfile and Compose include a health check endpoint (`/health`) that checks Redis connectivity.
- **Ports:** Exposes port 5000 for the Flask API, 6379 for Redis.
- **Volumes:** Redis data is persisted using a named Docker volume.
- **Dependencies:** Only Flask and redis-py are installed, minimizing the attack surface.
- **Docker Compose:** Orchestrates both the Flask API and Redis, ensuring the API waits for Redis to be available.

### How it fulfills the requirements
- **Flask API:** Implements endpoints to set/get key-value pairs in Redis, and a health check.
- **Redis Backend:** Uses the official Redis image as a backend, with persistent storage.
- **Security:** Runs as non-root, minimizes image size, and uses health checks.
- **Ease of Use:** Simple setup with Docker Compose, clear instructions, and persistent data.
