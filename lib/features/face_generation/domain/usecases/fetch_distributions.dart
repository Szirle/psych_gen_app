import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';
import 'package:psych_gen_app/features/face_generation/domain/repositories/distributions_repository.dart';

class FetchDistributionsUseCase {
  final DistributionsRepository repository;

  FetchDistributionsUseCase({required this.repository});

  Future<Map<ManipulatedDimensionName, List<double>>> call({
    Map<ManipulatedDimensionName, List<double>>? filters,
    int numPoints = 100,
    List<ManipulatedDimensionName>? variables,
  }) {
    return repository.fetchDistributions(
      filters: filters,
      numPoints: numPoints,
      variables: variables,
    );
  }
}
