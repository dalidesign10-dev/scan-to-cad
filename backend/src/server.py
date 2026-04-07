"""JSON-RPC 2.0 stdio server for the scan-to-CAD pipeline."""
import sys
import json
import traceback
from rpc_dispatcher import RPCDispatcher

dispatcher = RPCDispatcher()


def send_response(response: dict):
    line = json.dumps(response, default=str)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def send_progress(stage: str, pct: int, message: str = ""):
    send_response({
        "jsonrpc": "2.0",
        "method": "progress",
        "params": {"stage": stage, "pct": pct, "message": message}
    })


def handle_request(request: dict):
    method = request.get("method")
    params = request.get("params", {})
    req_id = request.get("id")

    try:
        result = dispatcher.dispatch(method, params, progress_callback=send_progress)
        if req_id is not None:
            send_response({"jsonrpc": "2.0", "id": req_id, "result": result})
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        if req_id is not None:
            send_response({
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32000, "message": str(e)}
            })


def main():
    sys.stderr.write("Python backend started\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            handle_request(request)
        except json.JSONDecodeError as e:
            send_response({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"}
            })


if __name__ == "__main__":
    main()
