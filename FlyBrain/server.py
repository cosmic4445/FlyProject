import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from brain import build_synthetic, load_csv
from controller import Fly


def make_handler(fly):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/status":
                self._send(200, fly._stats())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
                data = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/step":
                    self._send(200, fly.step(data.get("s", []), float(data.get("dx", 0.0))))
                elif self.path == "/start":
                    stats = fly.start_attempt()
                    print(f"[attempt {stats['attempt']}] starting")
                    self._send(200, stats)
                elif self.path == "/end":
                    stats = fly.end_attempt(float(data["progress"]), int(data.get("level", 1)), bool(data.get("cleared", False)))
                    tag = "CLEARED" if data.get("cleared") else "died"
                    print(f"[attempt {stats['attempt']}] {tag} at progress {float(data['progress']):.3f} "
                          f"(best {stats['best']}, deaths {stats['deaths']}, clears {stats['clears']})")
                    self._send(200, stats)
                elif self.path == "/reward":
                    out = fly.taste(int(data.get("level", 1)), int(data.get("tier", 1)))
                    print(f"[reward] level {data.get('level')} tier {data.get('tier')}: "
                          f"{out['sugar_spikes']} sugar-neuron spikes")
                    self._send(200, out)
                else:
                    self._send(404, {"error": "not found"})
            except (KeyError, ValueError, TypeError) as e:
                self._send(400, {"error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--neurons", help="neurons.csv (id,role) for a real connectome")
    ap.add_argument("--synapses", help="synapses.csv (pre,post,weight)")
    ap.add_argument("--weight-scale", type=float, default=1.0)
    ap.add_argument("--state", default="fly_state.json", help="where to save progress + learned weights between runs")
    ap.add_argument("--no-learn", action="store_true", help="freeze synapse weights (pure flailing, no learning)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if bool(args.neurons) != bool(args.synapses):
        sys.exit("--neurons and --synapses go together")
    plastic = not args.no_learn
    brain = (load_csv(args.neurons, args.synapses, args.weight_scale, args.seed, plastic=plastic) if args.neurons
             else build_synthetic(seed=args.seed, plastic=plastic))
    fly = Fly(brain, state_path=args.state)
    print(f"fly brain online: {brain.n} neurons, {brain.W_csr.nnz} synapses, {fly.C} sensory channels, "
          f"learning {'ON' if plastic else 'OFF'}")
    print(f"listening on http://{args.host}:{args.port}")
    try:
        ThreadingHTTPServer((args.host, args.port), make_handler(fly)).serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
