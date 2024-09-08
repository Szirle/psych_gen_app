import 'package:psych_gen_app/model/manipulated_dimension_name.dart';

class ManipulatedDimension {
   ManipulatedDimensionName name;
   double strength;
   int nLevels;

  ManipulatedDimension({required this.name, required this.strength, required this.nLevels});

  Map<String, dynamic> toJson() => {
        'name': name.toString().split('.').last,
        'strength': strength,
        'n_levels': nLevels,
      };
}
