# SAR.AI

Modified Multi-Level Pattern Histograms (MLPH) for SAR image classification.

Some results for different regions in and around New York City area:

![](./report_images/34.png)

![](./report_images/40.png)

![](./report_images/33.png)

## Methodology

Following the research on MLPH, we derive a pattern matrix for each pixel based on a threshold value, use these matrices varying bin lengths to get local pattern sub-histograms, concatenated to give local pattern histogram. With multiple thresholds, the concatenation of local pattern histograms gives MLPH for each pixel.

![](./report_images/1.png)

![](./report_images/2.png)

![](./report_images/3.png)
