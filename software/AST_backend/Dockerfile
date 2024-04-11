# Step 1: Build stage
# Using an Eclipse Temurin JDK image based on Alpine Linux
FROM eclipse-temurin:17-jdk-alpine as build
WORKDIR /workspace/app

# Copy the Gradle wrapper script and properties separately to leverage Docker cache
# This ensures that Gradle dependencies are only re-fetched if these files change
COPY gradlew gradle.properties* ./

# Copy Gradle wrapper files
COPY gradle gradle

# Adjust line endings and make the Gradle wrapper script executable
# This is crucial for environments where the script was edited on Windows
RUN sed -i 's/\r$//' gradlew && chmod +x ./gradlew

# Now, copy the rest of the project
COPY build.gradle settings.gradle ./
COPY src src

# Execute the build, skipping tests to speed up the build process
RUN ./gradlew build -x test

# Step 2: Package stage
# Switch to an Eclipse Temurin JRE image for a smaller final image
FROM eclipse-temurin:17-jre-alpine
VOLUME /tmp
WORKDIR /app

# Copy the built artifact from the build stage
COPY --from=build /workspace/app/build/libs/*.jar app.jar

# Define the command to run the application
ENTRYPOINT ["java","-jar","app.jar"]
