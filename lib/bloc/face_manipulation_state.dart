import 'dart:typed_data';
import 'package:equatable/equatable.dart';

abstract class FaceManipulationState extends Equatable {
  const FaceManipulationState();

  @override
  List<Object> get props => [];
}

class FaceManipulationInitial extends FaceManipulationState {}

class FaceManipulationLoading extends FaceManipulationState {}

class FaceManipulationLoaded extends FaceManipulationState {
  final List<Uint8List> images;

  const FaceManipulationLoaded(this.images);

  @override
  List<Object> get props => [images];
}

class FaceManipulationError extends FaceManipulationState {
  final String message;

  const FaceManipulationError(this.message);

  @override
  List<Object> get props => [message];
}
