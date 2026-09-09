import 'dart:typed_data';

import 'manipulated_dimension_name.dart';

/// Sampling criteria stay independent of the per-image experimental design.
/// Future text-ranking criteria belong in this object and its `selection` JSON.
class StimuliSelectionCriteria {
  final Map<ManipulatedDimensionName, List<double>> filters;

  StimuliSelectionCriteria(
      {Map<ManipulatedDimensionName, List<double>> filters = const {}})
      : filters = Map.unmodifiable(filters.map(
            (key, range) => MapEntry(key, List<double>.unmodifiable(range))));

  Map<String, dynamic> toJson() => {
        'filters': filters.map((key, range) => MapEntry(key.name, range)),
      };
}

class StimuliSelectionRequest {
  static const gridSide = 3;
  static const sampleCount = gridSide * gridSide;
  final int previewRevision;
  final double truncationPsi;
  final StimuliSelectionCriteria selection;

  StimuliSelectionRequest(
      {required this.previewRevision,
      required this.truncationPsi,
      required this.selection});

  Map<String, dynamic> toJson() => {
        'preview_revision': previewRevision,
        'sample_count': sampleCount,
        'truncation_psi': truncationPsi,
        'selection': selection.toJson(),
      };
}

class StimuliSelectionPreview {
  final List<Uint8List> images;
  final int previewRevision;
  final String source;
  final int? eligibleCount;
  final bool sampledWithReplacement;

  StimuliSelectionPreview(
      {required List<Uint8List> images,
      required this.previewRevision,
      required this.source,
      required this.eligibleCount,
      required this.sampledWithReplacement})
      : images = List.unmodifiable(images);
}
