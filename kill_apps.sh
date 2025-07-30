#!/bin/bash

uvicorn_pid=$(ps -ef | grep uvicorn | grep -v grep | awk '{print $2}')
echo "uvicorn pid: $uvicorn_pid"

npm_pid=$(ps -ef | grep npm | grep -v grep | awk '{print $2}')
echo "npm pid: $npm_pid"

kill $uvicorn_pid $npm_pid
