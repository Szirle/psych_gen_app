import 'dart:typed_data';
import 'package:equatable/equatable.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/face_manipulation_request.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension.dart';

class FaceImageGrid extends Equatable {
  final List<Uint8List> images;
  final List<String> dimensionNames;
  final List<List<double>> levelValues;

  const FaceImageGrid({
    required this.images,
    required this.dimensionNames,
    required this.levelValues,
  });

  factory FaceImageGrid.fromRequest(
    List<Uint8List> images,
    FaceManipulationRequest request,
  ) {
    return FaceImageGrid(
      images: images,
      dimensionNames: request.manipulatedDimensions
          .map((dimension) => dimension.apiName)
          .toList(),
      levelValues: request.manipulatedDimensions
          .map((dimension) => _levelsForDimension(dimension))
          .toList(),
    );
  }

  Uint8List? imageFor(
    List<ManipulatedDimension> targetDimensions,
    List<int> targetCoordinates,
  ) {
    if (targetDimensions.length != targetCoordinates.length || images.isEmpty) {
      return null;
    }

    final targetNames =
        targetDimensions.map((dimension) => dimension.apiName).toList();
    if (dimensionNames.length == targetNames.length &&
        dimensionNames.every(targetNames.contains)) {
      final oldCoordinates = <int>[];
      for (int oldAxis = 0; oldAxis < dimensionNames.length; oldAxis++) {
        final targetAxis = targetNames.indexOf(dimensionNames[oldAxis]);
        if (targetAxis < 0) return null;
        final targetCoordinate = targetCoordinates[targetAxis];
        final targetLevels = _levelsForDimension(targetDimensions[targetAxis]);
        if (targetCoordinate >= targetLevels.length) return null;

        final oldLevels = levelValues[oldAxis];
        int oldCoordinate;
        if (oldLevels.length == targetLevels.length) {
          oldCoordinate = targetCoordinate;
        } else {
          final targetValue = targetLevels[targetCoordinate];
          oldCoordinate = oldLevels.indexWhere(
            (value) => (value - targetValue).abs() < 0.000001,
          );
          if (oldCoordinate < 0) return null;
        }
        oldCoordinates.add(oldCoordinate);
      }
      final index =
          _flatIndex(oldCoordinates, levelValues.map((e) => e.length).toList());
      return index < images.length ? images[index] : null;
    }

    final targetCounts =
        targetDimensions.map((dimension) => dimension.nLevels).toList();
    final fallbackIndex = _flatIndex(targetCoordinates, targetCounts);
    return fallbackIndex < images.length ? images[fallbackIndex] : null;
  }

  static List<double> _levelsForDimension(ManipulatedDimension dimension) {
    if (dimension.nLevels <= 1) return [0.0];
    final step = (2 * dimension.strength) / (dimension.nLevels - 1);
    return List<double>.generate(
      dimension.nLevels,
      (index) => -dimension.strength + step * index,
    );
  }

  static int _flatIndex(List<int> coordinates, List<int> counts) {
    int index = 0;
    int stride = 1;
    for (int axis = 0; axis < coordinates.length; axis++) {
      index += coordinates[axis] * stride;
      stride *= counts[axis];
    }
    return index;
  }

  @override
  List<Object> get props => [images, dimensionNames, levelValues];
}

abstract class FaceManipulationState extends Equatable {
  const FaceManipulationState();

  @override
  List<Object?> get props => [];
}

class FaceManipulationInitial extends FaceManipulationState {}

class FaceManipulationLoading extends FaceManipulationState {
  final FaceImageGrid? previousGrid;

  const FaceManipulationLoading({this.previousGrid});

  @override
  List<Object?> get props => [previousGrid];
}

class FaceManipulationLoaded extends FaceManipulationState {
  final FaceImageGrid grid;

  const FaceManipulationLoaded(this.grid);

  List<Uint8List> get images => grid.images;

  @override
  List<Object?> get props => [grid];
}

class FaceManipulationError extends FaceManipulationState {
  final String message;
  final FaceImageGrid? previousGrid;

  const FaceManipulationError(this.message, {this.previousGrid});

  @override
  List<Object?> get props => [message, previousGrid];
}
