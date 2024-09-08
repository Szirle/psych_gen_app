import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:psych_gen_app/model/face_manipulation_request.dart';

class ApiService {
  static const String postRoute = '/images';    // Route for connection test

  Future<List<Uint8List>> postFaceManipulation(FaceManipulationRequest requestBody) async {
    try {
      String requestBodyJson = json.encode(requestBody.toJson());
      print("Request Body: $requestBodyJson");

      final response = await http.post(
        Uri.parse(postRoute),
        headers: {'Content-Type': 'application/json'},
        body: requestBodyJson,
      );

      if (response.statusCode == 200) {
        List<dynamic> responseList = json.decode(response.body);

        List<Uint8List> images = responseList.map((imageData) {
          return base64Decode(imageData);
        }).toList();

        return images;
      } else {
        print("Response Body: ${response.body}");
        throw Exception('ailed with status code ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('An error occurred during the request: $e');
    }
  }
}
