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
  Future<List<Uint8List>> postFaceManipulation(FaceManipulationRequest requestBody) async {
    try {
      // Print the request body
      String requestBodyJson = json.encode(requestBody.toJson());
      print("Request Body: $requestBodyJson");

      final response = await http.post(
        Uri.parse(postRoute),
        headers: {'Content-Type': 'application/json'},
        body: requestBodyJson, // Use the printed requestBodyJson
      );

      if (response.statusCode == 200) {
        // Parse the response as a list of base64-encoded image strings
        List<dynamic> responseList = json.decode(response.body);

        // Convert each image from base64 to Uint8List
        List<Uint8List> images = responseList.map((imageData) {
          return base64Decode(imageData);
        }).toList();

        return images;
      } else {
        // Print error details when the status code is not 200
        print("Error: Failed with status code ${response.statusCode}");
        print("Response Body: ${response.body}");
        throw Exception('Failed to post face manipulation request');
      }
    } catch (e) {
      // Print any caught errors
      print("Exception occurred: $e");
      throw Exception('An error occurred during the request: $e');
    }
  }
}
