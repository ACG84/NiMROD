#!/usr/bin/env python3
"""Drive the Colab CLI's interactive OAuth prompt through a pseudo-terminal.

The CLI prints a consent URL, then blocks reading an authorization code from
stdin.  Feeding that prompt from a pipe or a FIFO is unreliable: any transient
end-of-file makes the prompt abort ("Aborted.") and the PKCE verifier — which
only exists inside that process — is lost with it, invalidating the URL.

A pty behaves like a real terminal, so the process waits indefinitely.  This
driver runs in the background, publishes the URL to a file as soon as it
appears, and watches a second file for the code to submit.

    python scripts/colab_pty_auth.py &
    cat /tmp/colab_url.txt              # give this to the human
    echo "<code>" > /tmp/colab_code.txt # driver submits it to the prompt
"""

from __future__ import annotations

import os
import pty
import re
import select
import subprocess
import time

ROOT = "/home/user/NiMROD"
URL_FILE = "/tmp/colab_url.txt"
CODE_FILE = "/tmp/colab_code.txt"
LOG_FILE = "/tmp/colab_auth.log"
TIMEOUT_SECONDS = 3600

COMMAND = [
    f"{ROOT}/bin/micromamba", "run", "-n", "colabcli",
    "colab", "--auth=oauth2", "sessions",
]


def main() -> int:
    env = dict(os.environ, MAMBA_ROOT_PREFIX=f"{ROOT}/.mamba", TERM="dumb")

    master, slave = pty.openpty()
    proc = subprocess.Popen(
        COMMAND, stdin=slave, stdout=slave, stderr=slave,
        env=env, close_fds=True, preexec_fn=os.setsid,
    )
    os.close(slave)

    buffer = ""
    submitted = False
    url_written = False
    deadline = time.time() + TIMEOUT_SECONDS

    with open(LOG_FILE, "w") as log:
        while time.time() < deadline:
            readable, _, _ = select.select([master], [], [], 1.0)
            if readable:
                try:
                    chunk = os.read(master, 4096).decode("utf-8", "replace")
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                log.write(chunk)
                log.flush()
                if not url_written:
                    match = re.search(r"https://accounts\.google\.com/\S+", buffer)
                    if match:
                        with open(URL_FILE, "w") as fh:
                            fh.write(match.group(0))
                        url_written = True

            if not submitted and os.path.exists(CODE_FILE):
                code = open(CODE_FILE).read().strip()
                if code:
                    time.sleep(0.5)
                    os.write(master, (code + "\n").encode())
                    submitted = True
                    log.write(f"\n[driver] submitted code ({len(code)} chars)\n")
                    log.flush()

            if proc.poll() is not None and not select.select([master], [], [], 0)[0]:
                break

        log.write(f"\n[driver] child exit status: {proc.poll()}\n")
        log.flush()
    return proc.poll() or 0


if __name__ == "__main__":
    raise SystemExit(main())
