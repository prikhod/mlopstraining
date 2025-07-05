export RUN_ID=0b7cff6263b54203b737a279908b7f0d

python3 src/features/serve.py

curl -X POST -H 'Content-Type: application/json' -d '{"trip_distance":1}'  http://127.0.0.1:9696/predict 

{
  "duration": 39.75416552068733,
  "model_version": "0b7cff6263b54203b737a279908b7f0d"
}

zx
dsad
