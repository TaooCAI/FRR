# Full resolution in Stereo Matching

Due to the limitation of the computational resource, CNN based Stereo Matching often needs to downsample the input stereo pairs. However, as a relatively low-level task, Stereo Matching needs to output the disparity with the same size as its input images. This is the full resolution problem.

In experiments, I reproduced some others work, [PSMNet][] and [StereoNet](#references). Codes in PSMNet-FRR are partly based on the [PSMNet][] repository. Codes in StereoNet-SRR are written by myself.

## Requirements

* Python 3.6
* PyTorch 0.4.0

## Demos

**<p align="center">Image</p>**

<p align="center">
<img src="./PSMNet-FRR/demo_g32_retrain_from_scratch_using_flyingthings3D_TEST/7/2/0007.png" alt="image" height="60%" width="60%">
</p>

**<p align="center">Groundtruth Disparity</p>**

<p align="center">
<img align="center" alt="groundtruth" src="./PSMNet-FRR/demo_g32_retrain_from_scratch_using_flyingthings3D_TEST/7/2/gt.png" height="60%" width="60%">
</p>

**<p align="center">Prediction Disparity</p>**

<p align="center">
<img align="center" alt="prediction" src="./PSMNet-FRR/demo_g32_retrain_from_scratch_using_flyingthings3D_TEST/7/2/prediction.png" height="60%" width="60%">
</p>

---

## References

* Khamis S, Fanello S R, Rhemann C, et al. StereoNet: Guided Hierarchical Refinement for Real-Time Edge-Aware Depth Prediction[C]. european conference on computer vision, 2018: 596-613.
* Chang J, Chen Y. Pyramid Stereo Matching Network[C]. computer vision and pattern recognition, 2018: 5410-5418.

[PSMNet]: https://github.com/JiaRenChang/PSMNet
