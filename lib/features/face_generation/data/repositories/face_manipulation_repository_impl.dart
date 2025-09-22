import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:psych_gen_app/features/face_generation/data/datasources/face_manipulation_api_datasource.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/face_manipulation_request.dart';
import 'package:psych_gen_app/features/face_generation/domain/repositories/face_manipulation_repository.dart';

class FaceManipulationRepositoryImpl implements FaceManipulationRepository {
  final FaceManipulationApiDataSource _apiDataSource;

  FaceManipulationRepositoryImpl({FaceManipulationApiDataSource? apiDataSource})
      : _apiDataSource = apiDataSource ?? FaceManipulationApiDataSource();

  @override
  Future<List<Uint8List>> getFaceImages(FaceManipulationRequest request) async {
    if (kDebugMode) {
      return _generateMockImages(request);
    } else {
      return _apiDataSource.postFaceManipulation(request);
    }
  }

  Future<List<Uint8List>> _generateMockImages(
      FaceManipulationRequest request) async {
    int totalImages = 1;
    for (var dimension in request.manipulatedDimensions) {
      totalImages *= dimension.nLevels;
    }

    List<Uint8List> images = [];

    for (int i = 0; i < totalImages; i++) {
      final bytes = _transparentPng1x1;
      images.add(bytes);
    }

    return images;
  }

  // 1x1 transparent PNG (base64)
  static final Uint8List _transparentPng1x1 = base64Decode(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=');
}
