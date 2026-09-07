import 'dart:typed_data';

class FullResolutionFace {
  final Uint8List imageBytes;
  final Uint8List latentNpyBytes;

  const FullResolutionFace({
    required this.imageBytes,
    required this.latentNpyBytes,
  });
}

class FullResolutionResponse {
  final FullResolutionFace face;
  final List<int> selectedLevels;
  final String cacheKey;
  final bool previewIsLowResolution;

  const FullResolutionResponse({
    required this.face,
    required this.selectedLevels,
    required this.cacheKey,
    required this.previewIsLowResolution,
  });
}

class FullResolutionBatchUpdate {
  final List<FullResolutionResponse> items;
  final int cursor;
  final bool complete;
  final String? error;

  const FullResolutionBatchUpdate({
    required this.items,
    required this.cursor,
    required this.complete,
    this.error,
  });
}
