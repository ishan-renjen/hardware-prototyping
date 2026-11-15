"""Fetches a multiple sequence alignment (MSA) for one protein from the public
ColabFold MMseqs2 server, for AlphaFold inputs.

python alphafold_msa.py ubiquitin MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG

The sequence is sent to a third-party service (api.colabfold.com), so use this for
public proteins only. Results are cached in alphafold_inputs/<name>/, so the server
is contacted once per protein. The service asks clients to identify themselves; set
COLABFOLD_USER_AGENT (e.g. to a project name and contact address) to do so.
"""
import io
import json
import os
import sys
import tarfile
import time
import urllib.parse
import urllib.request

HOST = "https://api.colabfold.com"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(REPO, "alphafold_inputs")
# Alignments the server returns for mode "env": UniRef30, and environmental databases.
A3M_FILES = ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m")
QUERY_ID = 101  # how the server labels the (single) query in its results


def _request(path, data=None, retries=5):
    headers = {"User-Agent": os.environ.get("COLABFOLD_USER_AGENT", "hardware-prototyping-cosim")}
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{HOST}/{path}", data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except OSError as e:
            if attempt == retries - 1:
                raise
            print(f"  request to {path} failed ({e}); retrying")
            time.sleep(5)


def _status(raw):
    try:
        return json.loads(raw)
    except ValueError:
        return {"status": "ERROR", "message": raw[:200].decode(errors="replace")}


def fetch_msa(name, sequence):
    """Returns paths to the a3m files for this protein, fetching them if not cached."""
    sequence = sequence.strip().upper()
    out_dir = os.path.join(INPUTS, name)
    paths = [os.path.join(out_dir, f) for f in A3M_FILES]
    if all(os.path.exists(p) for p in paths):
        print(f"using cached MSA in {out_dir}")
        return paths
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "query.fasta"), "w") as f:
        f.write(f">{name}\n{sequence}\n")

    print(f"sending {name} ({len(sequence)} residues) to the ColabFold MSA server at {HOST}")
    ticket = _status(_request("ticket/msa", {"q": f">{QUERY_ID}\n{sequence}\n", "mode": "env"}))
    while ticket.get("status") in ("UNKNOWN", "RATELIMIT"):
        time.sleep(10)
        ticket = _status(_request("ticket/msa", {"q": f">{QUERY_ID}\n{sequence}\n", "mode": "env"}))
    while ticket.get("status") in ("PENDING", "RUNNING"):
        print(f"  {ticket['status'].lower()}...")
        time.sleep(10)
        ticket = _status(_request(f"ticket/{ticket['id']}"))
    if ticket.get("status") != "COMPLETE":
        raise RuntimeError(f"MSA server: {ticket}")

    archive = _request(f"result/download/{ticket['id']}")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name in A3M_FILES:
                text = tar.extractfile(member).read().decode()
                # Queries are separated by NUL bytes; there is only one query here.
                with open(os.path.join(out_dir, member.name), "w") as f:
                    f.write(text.replace("\x00", ""))
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"MSA server result lacked {missing}")
    for p in paths:
        hits = sum(1 for line in open(p) if line.startswith(">"))
        print(f"  {os.path.basename(p)}: {hits} sequences")
    return paths


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    fetch_msa(sys.argv[1], sys.argv[2])
