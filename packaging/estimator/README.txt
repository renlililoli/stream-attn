SeqAttn H3 Estimator - Windows x64

Extract the ZIP, then double-click seqattn-estimator.exe.
The application opens your default browser at a local 127.0.0.1 address.
No Python, PyTorch, CUDA, GPU, or internet connection is required to simulate.
Keep the console window open while using the page. Close it or press Ctrl+C
in that window to stop the server. Closing a browser tab does not stop it.
If the browser does not open, copy the address printed in the console.

Optional command-line usage:
  seqattn-estimator.exe --no-browser --port 8765

This is an unsigned application. Only run a build from a source you trust.
The executable extracts its bundled runtime to a temporary directory on startup.
The model estimates execution; it does not execute a DiT block or perform CUDA
calibration. Import measured device profiles through the web page instead.
Parameters and profiles can be exported from the page. The default port is
chosen automatically, so browser local storage may differ between launches.
Use --port 8765 for a stable browser storage origin when that port is available.

Source: renlililoli/stream-attn
See LICENSE.txt for the SeqAttn Apache-2.0 license and PYTHON-LICENSE.txt for Python.
