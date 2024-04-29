# Alternative (new?) Docker Support

This directory provides an experimental alternative to the current recommended approach.  It uses Docker Compose for development.  Instead of using `run_container.sh`, to start the container:

```
docker compose up
```

There is also a dockerfile.prod that can be used to build foundationpose containers that are ready to use and don't require executing build_all.sh