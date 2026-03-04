from server.main import app
import uvicorn
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11222)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
