class ObsoletePreviewException implements Exception {
  final String message;
  const ObsoletePreviewException(
      [this.message = 'This request belongs to an obsolete preview.']);

  @override
  String toString() => message;
}
