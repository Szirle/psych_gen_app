import 'package:psych_gen_app/model/manipulated_dimension_name.dart';

class ManipulatedDimension {
  final ManipulatedDimensionName? name;
  final double? strength;
  final int? nLevels;

  ManipulatedDimension({this.name, this.strength, this.nLevels});

  Map<String, dynamic> toJson() => {
        'name': name?.toString().split('.').last,
        'strength': strength,
        'n_levels': nLevels,
      };
}
