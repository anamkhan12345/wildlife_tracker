cd web_app
echo "STARTING API"
uvicorn api_app:app --host 0.0.0.0 --port 8000 & 
cd app || exit

echo "STARTING FRONTEND"
npm run dev -- --host=0.0.0.0 & 
