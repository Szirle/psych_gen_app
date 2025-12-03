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
        final List<Uint8List> images = <Uint8List>[];

        if (decoded is List) {
          if (decoded.isNotEmpty && decoded.first is List) {
            final firstLevel = decoded;
            final bool is3D = firstLevel.isNotEmpty &&
                firstLevel.first is List &&
                (firstLevel.first as List).isNotEmpty &&
                (firstLevel.first as List).first is List;

            if (is3D) {
              final l0 = firstLevel;
              final int n0 = l0.length;
              final int n1 = (l0.first as List).length;
              final int n2 = ((l0.first as List).first as List).length;

              for (int k = 0; k < n2; k++) {
                for (int j = 0; j < n1; j++) {
                  for (int i = 0; i < n0; i++) {
                    final dynamic imageData = (l0[i] as List)[j][k];
                    if (imageData is String) {
                      images.add(base64Decode(imageData));
                    }
                  }
                }
              }
            } else {
              for (final row in firstLevel) {
                if (row is List) {
                  for (final imageData in row) {
                    if (imageData is String) {
                      images.add(base64Decode(imageData));
                    }
                  }
                }
              }
            }
          } else {
            for (final imageData in decoded) {
              if (imageData is String) {
                images.add(base64Decode(imageData));
              }
            }
          }
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
        developer.log(
          'POST $postRoute | failure status=${response.statusCode} | body=${truncateForLog(response.body)}',
          name: _logName,
          level: 1000,
        );
        throw Exception(
            'Request failed with status code ${response.statusCode}');
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
