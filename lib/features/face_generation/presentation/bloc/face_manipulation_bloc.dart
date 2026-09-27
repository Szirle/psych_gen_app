import 'package:bloc/bloc.dart';
import 'package:psych_gen_app/core/errors/exceptions.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/face_manipulation_request.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension.dart';
import 'package:psych_gen_app/features/face_generation/domain/usecases/generate_face_images.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_event.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_state.dart';

class FaceManipulationBloc
    extends Bloc<FaceManipulationEvent, FaceManipulationState> {
  final GenerateFaceImagesUseCase _generateFaceImages;
  static const int debounceDuration = 500;
  int _latestRequest = 0;

  FaceManipulationBloc({required GenerateFaceImagesUseCase generateFaceImages})
      : _generateFaceImages = generateFaceImages,
        super(FaceManipulationInitial()) {
    on<LoadFaceImages>(_onLoadFaceImages);
  }

  /// Invalidate synchronously when leaving this workspace, including debounce.
  void cancelPendingPreview() => _latestRequest++;

  Future<void> _onLoadFaceImages(
    LoadFaceImages event,
    Emitter<FaceManipulationState> emit,
  ) async {
    final requestId = ++_latestRequest;
    final request = _snapshotRequest(event.request);
    final previousGrid = switch (state) {
      FaceManipulationLoaded(:final grid) => grid,
      FaceManipulationLoading(:final previousGrid) => previousGrid,
      FaceManipulationError(:final previousGrid) => previousGrid,
      _ => null,
    };
    emit(FaceManipulationLoading(previousGrid: previousGrid));
    await Future<void>.delayed(
      const Duration(milliseconds: debounceDuration),
    );
    if (requestId != _latestRequest || emit.isDone) return;
    try {
      final images = await _generateFaceImages(request);
      if (requestId != _latestRequest || emit.isDone) return;
      emit(FaceManipulationLoaded(
        FaceImageGrid.fromRequest(images, request),
      ));
    } on ObsoletePreviewException {
      // Ignored: request was superseded by a newer preview revision.
      return;
    } catch (e) {
      if (requestId != _latestRequest || emit.isDone) return;
      emit(FaceManipulationError(
        'Error fetching images: $e',
        previousGrid: previousGrid,
      ));
    }
  }

  FaceManipulationRequest _snapshotRequest(FaceManipulationRequest request) {
    return FaceManipulationRequest(
      manipulatedDimensions: request.manipulatedDimensions
          .map(
            (dimension) => ManipulatedDimension(
              name: dimension.name,
              traversalIndex: dimension.traversalIndex,
              strength: dimension.strength,
              nLevels: dimension.nLevels,
              rangeStart: dimension.rangeStart,
              rangeEnd: dimension.rangeEnd,
            ),
          )
          .toList(),
      truncationPsi: request.truncationPsi,
      numFaces: request.numFaces,
      preserveIdentity: request.preserveIdentity,
      changeFace: request.changeFace,
      mode: request.mode,
      previewRevision: request.previewRevision,
      filters: request.filters?.map(
        (key, value) => MapEntry(key, List<double>.from(value)),
      ),
      controlledVariables: request.controlledVariables == null
          ? null
          : List.of(request.controlledVariables!),
    );
  }
}
