#!/bin/bash
dask-scheduler --port 0 --scheduler-file scheduler_vasp.json --interface em2 1>LOG 2>LOGERR
