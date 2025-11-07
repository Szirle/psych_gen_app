import 'dart:typed_data';
import 'package:psych_gen_app/features/face_generation/data/datasources/face_manipulation_api_datasource.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/face_manipulation_request.dart';
import 'package:psych_gen_app/features/face_generation/domain/repositories/face_manipulation_repository.dart';

class FaceManipulationRepositoryImpl implements FaceManipulationRepository {
  final FaceManipulationApiDataSource _apiDataSource;

  FaceManipulationRepositoryImpl({FaceManipulationApiDataSource? apiDataSource})
      : _apiDataSource = apiDataSource ?? FaceManipulationApiDataSource();

  @override
  Future<List<Uint8List>> getFaceImages(FaceManipulationRequest request) async {
    return _apiDataSource.postFaceManipulation(request);
  }
}
