import 'package:psych_gen_app/model/manipulated_dimension.dart';

class FaceManipulationRequest {
  List<ManipulatedDimension> manipulatedDimensions;
  double truncationPsi;
  int numFaces;
  int maxSteps;
  bool preserveIdentity;
  String mode;

  FaceManipulationRequest({
    required this.manipulatedDimensions,
    required this.truncationPsi,
    required this.numFaces,
    required this.maxSteps,
    required this.preserveIdentity,
    required this.mode,
  });

  Map<String, dynamic> toJson() => {
        'manipulated_dimensions': manipulatedDimensions.map((dim) => dim.toJson()).toList(),
        'truncation_psi': truncationPsi,
        'num_faces': numFaces,
        'max_steps': maxSteps,
        'preserve_identity': preserveIdentity,
        'mode': mode,
      };
}
