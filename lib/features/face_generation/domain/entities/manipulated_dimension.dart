import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';

class ManipulatedDimension {
  ManipulatedDimensionName name;
  int? traversalIndex;
  double strength;
  int nLevels;
  double rangeStart;
  double rangeEnd;

  ManipulatedDimension(
      {required this.name,
      this.traversalIndex,
      required this.strength,
      required this.nLevels,
      this.rangeStart = 0.0,
      this.rangeEnd = 1.0});

  String get apiName => traversalIndex == null
      ? name.toString().split('.').last
      : 'traversal-$traversalIndex';

  String get displayName =>
      traversalIndex == null ? name.name : 'Traversal $traversalIndex';

  Map<String, dynamic> toJson() => {
        'name': apiName,
        'strength': strength,
        'n_levels': nLevels,
        'range_start': rangeStart,
        'range_end': rangeEnd,
      };
}
