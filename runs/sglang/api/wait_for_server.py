from argparse import ArgumentParser

from sglang.utils import wait_for_server

parser = ArgumentParser()
parser.add_argument("--host", type=str, required=True)
parser.add_argument("--port", type=str, required=True)
args = parser.parse_args()

wait_for_server(f"http://{args.host}:{args.port}")

print(f"Server started on http://{args.host}:{args.port}")
