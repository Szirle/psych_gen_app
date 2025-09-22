import 'dart:async';
import 'package:bloc/bloc.dart';
import 'package:psych_gen_app/features/face_generation/domain/usecases/generate_face_images.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_event.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/face_manipulation_state.dart';

class FaceManipulationBloc
    extends Bloc<FaceManipulationEvent, FaceManipulationState> {
  final GenerateFaceImagesUseCase _generateFaceImages;
  Timer? _debounce;
  static const int debounceDuration = 500;

  FaceManipulationBloc({required GenerateFaceImagesUseCase generateFaceImages})
      : _generateFaceImages = generateFaceImages,
        super(FaceManipulationInitial()) {
    on<LoadFaceImages>(_onLoadFaceImages);
  }

  Future<void> _onLoadFaceImages(
    LoadFaceImages event,
    Emitter<FaceManipulationState> emit,
  ) async {
    _debounce?.cancel();
    final completer = Completer<void>();
    _debounce = Timer(Duration(milliseconds: debounceDuration), () {
      completer.complete();
    });
    await completer.future;
    emit(FaceManipulationLoading());
    try {
      final images = await _generateFaceImages(event.request);
      emit(FaceManipulationLoaded(images));
    } catch (e) {
      emit(FaceManipulationError('Error fetching images: $e'));
    }
  }

  @override
  Future<void> close() {
    _debounce?.cancel();
    return super.close();
  }
}
