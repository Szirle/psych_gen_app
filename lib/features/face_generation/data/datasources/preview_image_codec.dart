import 'dart:convert';
import 'dart:typed_data';

/// Converts the backend grid into the preview's row-major display order.
///
/// The 2D API is already shaped [dimension1][dimension0], so each response row
/// maps directly to a UI row. The 3D API retains model order
/// [dimension0][dimension1][dimension2] and is traversed with dimension 0
/// changing fastest within each depth slice.
List<Uint8List> decodeImageGridForDisplay(
  List<dynamic> grid,
  List<int> levelCounts,
) {
  if (levelCounts.isEmpty || levelCounts.length > 3) {
    throw const FormatException('Expected level counts for 1 to 3 dimensions.');
  }

  final coordinates = List<int>.filled(levelCounts.length, 0);
  final images = <Uint8List>[];

  if (levelCounts.length == 2) {
    final rows = levelCounts[1];
    final columns = levelCounts[0];
    if (grid.length != rows) {
      throw const FormatException(
          'Image response row count does not match the request.');
    }
    for (final row in grid) {
      if (row is! List || row.length != columns) {
        throw const FormatException(
            'Image response column count does not match the request.');
      }
      for (final value in row) {
        if (value is! String) {
          throw const FormatException(
              'Image response contains a non-string value.');
        }
        images.add(base64Decode(value));
      }
    }
    return images;
  }

  dynamic valueAtCoordinates() {
    dynamic value = grid;
    for (final coordinate in coordinates) {
      if (value is! List || coordinate < 0 || coordinate >= value.length) {
        throw const FormatException(
            'Image response shape does not match the request.');
      }
      value = value[coordinate];
    }
    return value;
  }

  void visit(int axis) {
    if (axis < 0) {
      final value = valueAtCoordinates();
      if (value is! String) {
        throw const FormatException(
            'Image response contains a non-string value.');
      }
      images.add(base64Decode(value));
      return;
    }
    for (int level = 0; level < levelCounts[axis]; level++) {
      coordinates[axis] = level;
      visit(axis - 1);
    }
  }

  visit(levelCounts.length - 1);
  return images;
}
