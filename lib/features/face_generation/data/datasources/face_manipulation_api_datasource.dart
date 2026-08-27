import 'dart:convert';
import 'dart:developer' as developer;
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:psych_gen_app/core/constants/api_config.dart';
import 'package:psych_gen_app/core/utils/logging.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/face_manipulation_request.dart';

class FaceManipulationApiDataSource {
  static const String _logName = 'FaceManipulationApiDataSource';
  static String get postRoute => ApiConfig.resolve('/images');

  Future<List<Uint8List>> postFaceManipulation(
      FaceManipulationRequest requestBody) async {
    try {
      final requestBodyJson = json.encode(requestBody.toJson());
      final stopwatch = Stopwatch()..start();
      developer.log(
        'POST $postRoute | payload=${truncateForLog(requestBodyJson)}',
        name: _logName,
      );

      final response = await http.post(
        Uri.parse(postRoute),
        headers: {'Content-Type': 'application/json'},
        body: requestBodyJson,
      );
      stopwatch.stop();

      developer.log(
        'POST $postRoute | status=${response.statusCode} | duration=${stopwatch.elapsedMilliseconds}ms | bodyLength=${response.body.length}',
        name: _logName,
      );

      if (response.statusCode == 200) {
        final dynamic decoded = json.decode(response.body);
        if (decoded is List) {
          final images = decodeImageGridForDisplay(
            decoded,
            requestBody.manipulatedDimensions
                .map((dimension) => dimension.nLevels)
                .toList(),
          );
          developer.log(
            'POST $postRoute | decodedImages=${images.length}',
            name: _logName,
          );
          return images;
        }

        final message =
            'Unexpected response format for $postRoute: ${truncateForLog(response.body)}';
        developer.log(
          message,
          name: _logName,
          level: 1000,
        );
        throw Exception(message);
      } else {
        String serverMessage = response.body;
        try {
          final dynamic body = json.decode(response.body);
          if (body is Map && body['error'] is String) {
            serverMessage = body['error'] as String;
          }
        } catch (_) {}
        developer.log(
          'POST $postRoute | failure status=${response.statusCode} | body=${truncateForLog(response.body)}',
          name: _logName,
          level: 1000,
        );
        throw Exception(
            'Request failed (${response.statusCode}): $serverMessage');
      }
    } catch (e, stack) {
      developer.log(
        'POST $postRoute | exception=$e',
        name: _logName,
        error: e,
        stackTrace: stack,
        level: 1000,
      );
      throw Exception('An error occurred during the request: $e');
    }
  }
}

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
        throw const FormatException('Image response contains a non-string value.');
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
