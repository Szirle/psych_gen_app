import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';

class ManipulatedDimension {
  ManipulatedDimensionName name;
  double strength;
  int nLevels;
  double rangeStart;
  double rangeEnd;

  ManipulatedDimension(
      {required this.name,
      required this.strength,
      required this.nLevels,
      this.rangeStart = 0.0,
      this.rangeEnd = 1.0});

  Map<String, dynamic> toJson() => {
        'name': name.toString().split('.').last,
        'strength': strength,
        'n_levels': nLevels,
        'range_start': rangeStart,
        'range_end': rangeEnd,
      };
}
