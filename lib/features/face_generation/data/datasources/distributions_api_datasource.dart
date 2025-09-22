import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';

class DistributionsApiDataSource {
  static const String _defaultBaseUrl = 'http://127.0.0.1:8000';
  static const String baseUrl =
      String.fromEnvironment('API_BASE_URL', defaultValue: _defaultBaseUrl);
  static String get postRoute => '$baseUrl/distributions';

  Future<Map<ManipulatedDimensionName, List<double>>> fetchDistributions({
    Map<ManipulatedDimensionName, List<double>>? filters,
    int numPoints = 100,
    List<ManipulatedDimensionName>? variables,
  }) async {
    final body = {
      if (filters != null)
        'filters': filters.map((k, v) => MapEntry(k.name, v)),
      'num_points': numPoints,
      if (variables != null) 'variables': variables.map((e) => e.name).toList(),
    };

    final response = await http.post(
      Uri.parse(postRoute),
      headers: {'Content-Type': 'application/json'},
      body: json.encode(body),
    );

    if (response.statusCode != 200) {
      throw Exception('Distributions request failed: ${response.statusCode}');
    }

    final Map<String, dynamic> decoded = json.decode(response.body);
    final Map<String, dynamic> dist =
        (decoded['distributions'] as Map<String, dynamic>? ?? {});

    final Map<ManipulatedDimensionName, List<double>> result = {};
    dist.forEach((key, value) {
      try {
        final enumKey = ManipulatedDimensionName.values
            .firstWhere((e) => e.name == key, orElse: () => throw '');
        final List<dynamic> arr = (value as List<dynamic>);
        result[enumKey] = arr.map((e) => (e as num).toDouble()).toList();
      } catch (_) {
        // ignore keys not present in enum
      }
    });
    return result;
  }
}
