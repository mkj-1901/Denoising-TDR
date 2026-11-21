# TDR live demo

1. Clone the authors' data: `git clone https://github.com/zimugh/Denoising-TDR`
2. Install the requirements: `pip install numpy scipy matplotlib`
3. Run: `python3 demo.py Denoising-TDR` (takes about 1 s on a laptop CPU)
4. Open `demo_result.png`.

The primary denoiser in this demo is a fast stand-in (a Gaussian filter), so it runs live without TensorFlow or a GPU. `restoration.py` is the same restoration code used in the report.
