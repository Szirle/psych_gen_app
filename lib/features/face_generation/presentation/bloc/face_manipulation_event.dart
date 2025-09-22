import 'package:equatable/equatable.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/face_manipulation_request.dart';

abstract class FaceManipulationEvent extends Equatable {
  const FaceManipulationEvent();

  @override
  List<Object> get props => [];
}

class LoadFaceImages extends FaceManipulationEvent {
  final FaceManipulationRequest request;

  const LoadFaceImages(this.request);

  @override
  List<Object> get props => [request];
}
