import 'package:psych_gen_app/model/manipulated_dimension.dart';

class FaceManipulationRequest {
  List<ManipulatedDimension> manipulatedDimensions;
  double truncationPsi;
  int numFaces;
  bool preserveIdentity;
  bool changeFace;
  String mode;

  FaceManipulationRequest({
    required this.manipulatedDimensions,
    required this.truncationPsi,
    required this.numFaces,
    required this.preserveIdentity,
    this.changeFace = false,
    required this.mode,
  });

  Map<String, dynamic> toJson() => {
        'manipulated_dimensions':
            manipulatedDimensions.map((dim) => dim.toJson()).toList(),
        'truncation_psi': truncationPsi,
        'num_faces': numFaces,
        'preserve_identity': preserveIdentity,
        'change_face': changeFace,
        'mode': mode,
      };
}
