import time
from argparse import ArgumentParser

import requests

parser = ArgumentParser()
parser.add_argument("--host", type=str, required=True)
parser.add_argument("--port", type=str, required=True)
args = parser.parse_args()

start_time = time.time()
response = requests.get(
    f"http://{args.host}:{args.port}/v1/models",
    headers={"Authorization": "Bearer None"},
)
if response.status_code == 200:
    time.sleep(5)
    print(
        """\n
        NOTE: Typically, the server runs in a separate terminal.
        In this notebook, we run the server and notebook code together, so their outputs are combined.
        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.
        We are running those notebooks in a CI parallel environment, so the throughput is not representative of the actual performance.
        """
    )

print(f"Server connected successfully on http://{args.host}:{args.port}")
