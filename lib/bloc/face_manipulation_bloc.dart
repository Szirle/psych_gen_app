import 'dart:async';
import 'package:bloc/bloc.dart';
import 'package:psych_gen_app/bloc/face_manipulation_event.dart';
import 'package:psych_gen_app/bloc/face_manipulation_state.dart';
import 'package:psych_gen_app/repository/face_manipulation_repository.dart';

class FaceManipulationBloc
    extends Bloc<FaceManipulationEvent, FaceManipulationState> {
  final FaceManipulationRepository _repository;
  Timer? _debounce;
  static const int debounceDuration = 500;

  FaceManipulationBloc({required FaceManipulationRepository repository})
      : _repository = repository,
        super(FaceManipulationInitial()) {
    on<LoadFaceImages>(_onLoadFaceImages);
  }

  Future<void> _onLoadFaceImages(
    LoadFaceImages event,
    Emitter<FaceManipulationState> emit,
  ) async {
    // Cancel previous debounce timer
    _debounce?.cancel();

    // Create new debounce timer
    final completer = Completer<void>();
    _debounce = Timer(Duration(milliseconds: debounceDuration), () {
      completer.complete();
    });

    // Wait for debounce
    await completer.future;

    emit(FaceManipulationLoading());

    try {
      final images = await _repository.getFaceImages(event.request);
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
