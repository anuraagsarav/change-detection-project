\# Change Detection in Satellite Images Using Deep Learning



\## Overview

This project focuses on detecting changes in multi-temporal satellite imagery using

both classical image processing techniques and deep learning–based models.

The work is structured into multiple phases, progressing from baseline methods

to advanced Siamese U-Net architectures.



\## Project Structure

\- \*\*Phase 1\*\*: Classical pixel-based change detection (baseline)

\- \*\*Phase 2\*\*: CNN-based feature extraction

\- \*\*Phase 3\*\*: Siamese CNN for change detection

\- \*\*Phase 4\*\*: Improved loss functions and optimization

\- \*\*Phase 5\*\*: Siamese U-Net architecture

\- \*\*Phase 6\*\*: Evaluation, optimization, and analysis



\## Dataset

This project uses publicly available satellite imagery datasets:



\- \*\*LEVIR-CD\*\*  

&nbsp; https://justchenhao.github.io/LEVIR/



\- \*\*Custom Google Earth Samples (for academic use)\*\*  

&nbsp; Data sourced manually for experimental comparison.



> \*\*Note:\*\* Datasets are \*\*not included\*\* in this repository due to size constraints.

> Please download them separately using the links above.



\## Setup Instructions

```bash

python -m venv venv

source venv/bin/activate   # Windows: venv\\Scripts\\activate

pip install -r requirements.txt



