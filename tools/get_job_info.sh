#!/bin/bash

# Usage: ./get_job_info.sh <JOB_ID>

JOB_ID="$1"

scontrol show job $JOB_ID | awk '
/UserId=/ {print "User:", $1}
/NodeList=/ {print "Nodes:", $1}
/StartTime=/ {print "Started:", $1}
/EndTime=/ {print "Ended:", $1}
/TimeLimit=/ {print "TimeLimit:", $1}
/TRES=/ {
  match($0, /cpu=([0-9]+)/, a); cpus=a[1];
  match($0, /mem=([0-9]+[A-Z]+)/, b); mem=b[1];
  match($0, /gres\/gpu=([0-9]+)/, c); gpus=c[1];
  if (!gpus) gpus=0;
  printf "CPUs: %s\nMemory: %s\nGPUs: %s\n", cpus, mem, gpus;
  exit  # exit after first match to avoid duplicates
}
'
