import 'dart:async';
import 'package:bloc/bloc.dart';
import '../../domain/entities/stimuli_selection_request.dart';

class StimuliSelectionState {
  final StimuliSelectionPreview? preview;
  final bool isLoading;
  final String? error;
  const StimuliSelectionState(
      {this.preview, this.isLoading = false, this.error});
}

/// Each request is immutable; only the latest submitted snapshot may publish.
/// Like manipulation previews, keep the previous image tiles during debounce
/// and loading. An error clears the sample so excluded faces cannot look valid.
class StimuliSelectionCubit extends Cubit<StimuliSelectionState> {
  final Future<StimuliSelectionPreview> Function(StimuliSelectionRequest)
      fetchPreview;
  final Duration debounce;
  Timer? _timer;
  int _generation = 0;

  StimuliSelectionCubit(
      {required this.fetchPreview,
      this.debounce = const Duration(milliseconds: 500)})
      : super(const StimuliSelectionState());

  void load(StimuliSelectionRequest request) {
    final generation = ++_generation;
    _timer?.cancel();
    emit(StimuliSelectionState(preview: state.preview, isLoading: true));
    _timer = Timer(debounce, () async {
      try {
        final preview = await fetchPreview(request);
        if (!isClosed && generation == _generation) {
          emit(StimuliSelectionState(preview: preview));
        }
      } catch (error) {
        if (!isClosed && generation == _generation) {
          emit(StimuliSelectionState(error: error.toString()));
        }
      }
    });
  }

  void cancelPending() {
    _timer?.cancel();
    _generation++;
    emit(StimuliSelectionState(preview: state.preview, error: state.error));
  }

  @override
  Future<void> close() {
    _timer?.cancel();
    _generation++;
    return super.close();
  }
}
