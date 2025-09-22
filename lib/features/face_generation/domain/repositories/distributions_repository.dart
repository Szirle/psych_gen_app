import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';

abstract class DistributionsRepository {
  Future<Map<ManipulatedDimensionName, List<double>>> fetchDistributions({
    Map<ManipulatedDimensionName, List<double>>? filters,
    int numPoints = 100,
    List<ManipulatedDimensionName>? variables,
  });
}
