#!/bin/bash
dask-scheduler --port 0 --scheduler-file scheduler_nwchem.json --interface em2 1>LOG 2>LOGERR
