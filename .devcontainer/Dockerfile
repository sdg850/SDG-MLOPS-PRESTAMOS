FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# This Python project does not use the obsolete Yarn APT repository.
# Remove it before Features run apt; retain signature verification everywhere.
RUN rm -f /etc/apt/sources.list.d/yarn.list
