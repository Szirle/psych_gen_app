import 'package:psych_gen_app/model/manipulated_dimension.dart';

class FaceManipulationRequest {
  final List<ManipulatedDimension>? manipulatedDimensions;
  final double? truncationPsi;
  final int? numFaces;
  final int? maxSteps;
  final bool? preserveIdentity;
  final String? mode;

  FaceManipulationRequest({
    this.manipulatedDimensions,
    this.truncationPsi,
    this.numFaces,
    this.maxSteps,
    this.preserveIdentity,
    this.mode,
  });

  Map<String, dynamic> toJson() => {
    'manipulated_dimensions':
    manipulatedDimensions?.map((dim) => dim.toJson()).toList(),
    'truncation_psi': truncationPsi,
    'num_faces': numFaces,
    'max_steps': maxSteps,
    'preserve_identiy': preserveIdentity,
    'mode': mode,
  };
}
