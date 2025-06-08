import 'dart:typed_data';
import 'package:psych_gen_app/model/face_manipulation_request.dart';

abstract class FaceManipulationRepository {
  Future<List<Uint8List>> getFaceImages(FaceManipulationRequest request);
}
