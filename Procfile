web: python bootstrap.py && gunicorn -w 1 -b 0.0.0.0:$PORT app_user:app
worker: python run_scanner.py
