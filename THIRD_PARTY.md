The files under `src/face_intel/contracts/` are unmodified generated artifacts,
source, and release metadata from [lost-rob0t/star-lang](https://github.com/lost-rob0t/star-lang)
at core commit `e9ec1d883627d186aafe0b3647bdc29baea03543` and extension commit
`cf7dbd357aeb557ed97435e3c06f1b7d367dbf50` (Star Language PR #201).

Their upstream AGPL-3.0-only license is retained as
`src/face_intel/contracts/LICENSE.star-lang`. `pin.json` is this consumer's
dependency metadata. `__init__.py` is this consumer's package marker.

The additive `org.starintel/face-intel@1` source lives under
`specs/starintel/extensions/face-intel/0.1.0/` upstream. Its vendored files under
`contracts/face_extension/` retain exact upstream bytes and have a separate pin.

This change does not select a separate project-wide license for face-intel.

Optional similarity uses installed OpenCV (Apache-2.0) and NumPy (BSD-3-Clause)
packages. Model weights are not vendored. The documented SFace model is from
[OpenCV Zoo at 47534e27c9851bb1128ccc0102f1145e27f23f98](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface),
whose model directory includes an Apache-2.0 LICENSE. Operators distributing the
weights must retain the upstream license and notices. Retrieved 2026-10-10.
