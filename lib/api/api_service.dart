import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:psych_gen_app/model/face_manipulation_request.dart';

class ApiService {
  static const String postRoute = '/images'; // Route for connection test

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
            // Nested list (e.g., 2D grid): flatten row-major
            for (final row in decoded) {
              if (row is List) {
                for (final imageData in row) {
                  if (imageData is String) {
                    images.add(base64Decode(imageData));
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
