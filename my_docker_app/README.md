# Docker Container Setup Instructions

1. **Clone or create the project directory and add the following files:**
   - `Dockerfile` (from above)
   - `package.json` (from above)
   - `server.js` (from above)
   - `docker-compose.yml` (from above)

2. **Build and start the containers:**
   ```sh
   docker-compose up --build
   ```

3. **Access the web app:**
   - Open your browser and go to [http://localhost:3000](http://localhost:3000)
   - You should see a greeting and a visit counter (stored in MongoDB)

4. **Stop the containers:**
   ```sh
   docker-compose down
   ```
   (Data is persisted in the `mongo_data` Docker volume.)


## Explanation

This setup uses a multi-stage Docker build for the Node.js app to keep the final image small and production-ready. The app runs as a non-root user for security. Environment variables are used for MongoDB connection details. The health check ensures the app is running and responsive. The `docker-compose.yml` file orchestrates both the Node.js app and a MongoDB service, with persistent storage for MongoDB data. The app exposes port 3000 and connects to MongoDB using the service name `mongo`. This setup is secure, efficient, and easy to run with a single `docker-compose up` command.