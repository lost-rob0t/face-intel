The files under `src/face_intel/contracts/` are unmodified generated artifacts,
source, and release metadata from [lost-rob0t/star-lang](https://github.com/lost-rob0t/star-lang)
at core commit `e9ec1d883627d186aafe0b3647bdc29baea03543`.

Their upstream AGPL-3.0-only license is retained as
`src/face_intel/contracts/LICENSE.star-lang`. `pin.json` is this consumer's
dependency metadata. `__init__.py` is this consumer's package marker.

This change does not select a separate project-wide license for face-intel.

Optional similarity uses installed OpenCV (Apache-2.0) and NumPy (BSD-3-Clause)
packages. Model weights are not vendored. The documented SFace model is from
[OpenCV Zoo at 47534e27c9851bb1128ccc0102f1145e27f23f98](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface),
whose model directory includes an Apache-2.0 LICENSE. Operators distributing the
weights must retain the upstream license and notices. Retrieved 2026-10-10.

Optional face extraction uses the YuNet 2023mar float32 model from the same pinned
[OpenCV Zoo commit](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet),
whose model directory retains an MIT LICENSE. Weights are downloaded separately,
not vendored. Retain its license and notices when distributing them. Retrieved
2026-10-10. The opt-in/CI inference fixture is OpenCV 4.13.0
`samples/data/lena.jpg`, downloaded outside this repository and checked by SHA-256.
It is used only to verify model execution, not as an accuracy benchmark.
