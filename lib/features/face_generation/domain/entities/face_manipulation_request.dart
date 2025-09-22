import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';

class FaceManipulationRequest {
  List<ManipulatedDimension> manipulatedDimensions;
  double truncationPsi;
  int numFaces;
  bool preserveIdentity;
  bool changeFace;
  String mode;
  Map<ManipulatedDimensionName, List<double>>? filters;

  FaceManipulationRequest({
    required this.manipulatedDimensions,
    required this.truncationPsi,
    required this.numFaces,
    required this.preserveIdentity,
    this.changeFace = false,
    required this.mode,
    this.filters,
  });

  Map<String, dynamic> toJson() => {
        'manipulated_dimensions':
            manipulatedDimensions.map((dim) => dim.toJson()).toList(),
        'truncation_psi': truncationPsi,
        'num_faces': numFaces,
        'preserve_identity': preserveIdentity,
        'change_face': changeFace,
        'mode': mode,
        if (filters != null && filters!.isNotEmpty)
          'filters': filters!.map((k, v) => MapEntry(k.name, v)),
      };
}
