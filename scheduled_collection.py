import subprocess
import time
from datetime import datetime, time as dtime

script_path = 'data_pipeline.py'
kill_time = dtime(20,0)

while True:
    now = datetime.now().time()
    
    if now > kill_time:
        print("Reached kill time - stopping data collection")
        break

    print("Running Data Collection.....")
    subprocess.run(["python", script_path])

    for _ in range(60):
        time.sleep(60)
        
        now = datetime.now().time()
        if now >= kill_time:
            print("Reached stop time during wait — exiting.")
            exit()