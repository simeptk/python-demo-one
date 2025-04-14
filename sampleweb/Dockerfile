FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build

# Set the working directory
WORKDIR /opt/app-root/src

# Copy the project files granting default user permissions
COPY --chown=1001:0 . .

# Restore dependencies
RUN dotnet restore

# Build the application
RUN dotnet build -c Release

# Stage 2: Publish the application
FROM build AS publish

# Publish the application
RUN dotnet publish -c Release -o /opt/app-root/src/publish

# Stage 3: Finalize the image
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS final

# Set the working directory
WORKDIR /opt/app-root/src

# Copy the published files from the previous stage
COPY --from=publish /opt/app-root/src/publish .

EXPOSE 8080
EXPOSE 8443

ENV ASPNETCORE_ENVIRONMENT=Development

# Set the entry point
ENTRYPOINT ["dotnet", "sampleweb.dll"] 
