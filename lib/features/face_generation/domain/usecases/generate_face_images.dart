import 'dart:typed_data';
import 'package:psych_gen_app/features/face_generation/domain/entities/face_manipulation_request.dart';
import 'package:psych_gen_app/features/face_generation/domain/repositories/face_manipulation_repository.dart';

class GenerateFaceImagesUseCase {
  final FaceManipulationRepository repository;

  GenerateFaceImagesUseCase({required this.repository});

  Future<List<Uint8List>> call(FaceManipulationRequest request) {
    return repository.getFaceImages(request);
  }
}
