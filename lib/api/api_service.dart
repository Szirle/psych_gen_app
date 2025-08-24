import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:psych_gen_app/model/face_manipulation_request.dart';

class ApiService {
  // Configure backend base URL. Override with --dart-define=API_BASE_URL=https://your-host
  static const String _defaultBaseUrl = 'http://127.0.0.1:8000';
  static const String baseUrl =
      String.fromEnvironment('API_BASE_URL', defaultValue: _defaultBaseUrl);
  static String get postRoute => '$baseUrl/images';

  Future<List<Uint8List>> postFaceManipulation(
      FaceManipulationRequest requestBody) async {
    try {
      String requestBodyJson = json.encode(requestBody.toJson());
      print("Request Body: $requestBodyJson");

      final response = await http.post(
        Uri.parse(postRoute),
        headers: {'Content-Type': 'application/json'},
        body: requestBodyJson,
      );

      if (response.statusCode == 200) {
        final dynamic decoded = json.decode(response.body);

        // Support both flat list ["b64", ...] and nested list [["b64", ...], ...]
        final List<Uint8List> images = <Uint8List>[];

        if (decoded is List) {
          if (decoded.isNotEmpty && decoded.first is List) {
            final firstLevel = decoded;
            // Detect 3D structure: [n0][n1][n2] where leaf is String
            final bool is3D = firstLevel.isNotEmpty &&
                firstLevel.first is List &&
                (firstLevel.first as List).isNotEmpty &&
                (firstLevel.first as List).first is List;

            if (is3D) {
              final l0 = firstLevel; // i0 dimension (X)
              final int n0 = l0.length;
              final int n1 = (l0.first as List).length;
              final int n2 = ((l0.first as List).first as List).length;

              // Desired flatten order: fastest n0 (X), then n1 (Y), slowest n2 (slider)
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
              // 2D: flatten row-major
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
            // Flat list
            for (final imageData in decoded) {
              if (imageData is String) {
                images.add(base64Decode(imageData));
              }
            }
          }
          return images;
        }

        // Fallback: unexpected structure
        throw Exception(
            'Unexpected response format: ${response.body.substring(0, response.body.length.clamp(0, 200))}');
      } else {
        print("Response Body: ${response.body}");
        throw Exception('ailed with status code ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('An error occurred during the request: $e');
    }
  }
}
