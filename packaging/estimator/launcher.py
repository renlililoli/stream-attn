"""Entry point for the standalone estimator (no GPU runtime required)."""

from seqattn_core.estimation.web.server import main

if __name__ == "__main__":
    main(default_port=0, default_open_browser=True)
