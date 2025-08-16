import subprocess
import time
import sys
from datetime import datetime, time as dtime, timedelta

script_path = 'data_pipeline.py'
kill_time = dtime(20,0)

while True:
    now = datetime.now().time()
    
    if now > kill_time:
        print("Reached kill time - stopping data collection")
        break

    # Run pipeline
    print("Running Data Collection....")
    stage1_start = time.time()
    subprocess.run([sys.executable, script_path, '--cam', '1', '--det_area', '20', '--det_limit', '1000'])
    stage1_end = time.time()
    stage1_dur = stage1_end - stage1_start

    # if stage1_dur > 3600:
    #     print(f"All Detection Time (mins): {stage1_dur / 60}, starting over" )
    # else:
    #     print(f"All Detection Time (mins): {stage1_dur / 60}, starting Large Detections Only" )

    #     # Setup for large detections only
    #     stage2_now = datetime.now()
    #     next_hour = (stage2_now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
        
    #     # Find the delta
    #     delta = next_hour - stage2_now
    #     print(f"Next hour mark: {next_hour}")
    #     print(f"Time remaining: {delta} ({delta.total_seconds()} seconds)")

    #     while datetime.now() < next_hour:
    #         print("Running Data Collection for large objects only.....")
    #         subprocess.run([sys.executable, script_path, '--cam', '1', '--det_area', '200', '--det_limit', '10'])

    #         # Kill if past 8 pm
    #         now = datetime.now().time()
    #         if now >= kill_time:
    #             print("Reached stop time during wait — exiting.")
    #             exit()