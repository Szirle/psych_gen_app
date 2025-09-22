import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:psych_gen_app/features/face_generation/domain/entities/face_manipulation_request.dart';

class FaceManipulationApiDataSource {
  static const String _defaultBaseUrl = 'http://127.0.0.1:8000';
  static const String baseUrl =
      String.fromEnvironment('API_BASE_URL', defaultValue: _defaultBaseUrl);
  static String get postRoute => '$baseUrl/images';

  Future<List<Uint8List>> postFaceManipulation(
      FaceManipulationRequest requestBody) async {
    try {
      String requestBodyJson = json.encode(requestBody.toJson());
      final response = await http.post(
        Uri.parse(postRoute),
        headers: {'Content-Type': 'application/json'},
        body: requestBodyJson,
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
          return images;
        }

        throw Exception(
            'Unexpected response format: ${response.body.substring(0, response.body.length.clamp(0, 200))}');
      } else {
        throw Exception('ailed with status code ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('An error occurred during the request: $e');
    }
  }
}
