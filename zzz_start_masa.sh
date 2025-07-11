cd /home/user2/wqq/masa
pkill -f "uvicorn z_masaserver:app"
# nohup python -m uvicorn z_masaserver:app --host 0.0.0.0 --port 10020 --workers 1 > /dev/null 2>&1 &
nohup python -m uvicorn z_masaserver:app --host 0.0.0.0 --port 10020 --workers 1 > z_masaserver_printlog.log 2>&1 &
