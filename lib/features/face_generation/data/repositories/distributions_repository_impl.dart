import 'package:psych_gen_app/features/face_generation/data/datasources/distributions_api_datasource.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';
import 'package:psych_gen_app/features/face_generation/domain/repositories/distributions_repository.dart';

class DistributionsRepositoryImpl implements DistributionsRepository {
  final DistributionsApiDataSource _api;
  DistributionsRepositoryImpl({DistributionsApiDataSource? api})
      : _api = api ?? DistributionsApiDataSource();

  @override
  Future<Map<ManipulatedDimensionName, List<double>>> fetchDistributions({
    Map<ManipulatedDimensionName, List<double>>? filters,
    int numPoints = 100,
    List<ManipulatedDimensionName>? variables,
  }) {
    return _api.fetchDistributions(
      filters: filters,
      numPoints: numPoints,
      variables: variables,
    );
  }
}
