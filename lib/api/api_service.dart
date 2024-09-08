import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:psych_gen_app/model/face_manipulation_request.dart';

class ApiService {
  static const String imageRoute = '/image';  // Route for fetching test image
  static const String testRoute = '/test';    // Route for connection test
  static const String postRoute = '/images';    // Route for connection test

  Future<String> fetchData() async {
    final response = await http.get(Uri.parse(testRoute));

    if (response.statusCode == 200) {
      return json.decode(response.body)['message'];
    } else {
      throw Exception('Failed to load data');
    }
  }

  Future<Uint8List> fetchImage() async {
    final response = await http.get(Uri.parse(imageRoute));

    if (response.statusCode == 200) {
      return response.bodyBytes;
    } else {
      throw Exception('Failed to load image');
    }
  }

  // Method to send POST request with FaceManipulationRequest
  Future<List<Uint8List>> postFaceManipulation(
      FaceManipulationRequest requestBody) async {
    final response = await http.post(
      Uri.parse(postRoute),
      headers: {'Content-Type': 'application/json'},
      body: json.encode(requestBody.toJson()),
    );

    if (response.statusCode == 200) {
      List<dynamic> responseList = json.decode(response.body);

      List<Uint8List> images = responseList.map((imageData) {
        return base64Decode(imageData);
      }).toList();

      return images;
    } else {
      throw Exception('Failed to post face manipulation request');
    }
  }
}
