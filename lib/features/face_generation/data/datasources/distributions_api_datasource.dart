import 'dart:convert';
import 'dart:developer' as developer;
import 'package:http/http.dart' as http;
import 'package:psych_gen_app/core/constants/api_config.dart';
import 'package:psych_gen_app/core/utils/logging.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';

class DistributionsApiDataSource {
  static const String _logName = 'DistributionsApiDataSource';
  static String get postRoute => ApiConfig.resolve('/distributions');

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

    final payload = json.encode(body);
    final stopwatch = Stopwatch()..start();
    developer.log(
      'POST $postRoute | payload=${truncateForLog(payload)}',
      name: _logName,
    );

    try {
      final response = await http.post(
        Uri.parse(postRoute),
        headers: {'Content-Type': 'application/json'},
        body: payload,
      );
      stopwatch.stop();

      developer.log(
        'POST $postRoute | status=${response.statusCode} | duration=${stopwatch.elapsedMilliseconds}ms | bodyLength=${response.body.length}',
        name: _logName,
      );

      if (response.statusCode != 200) {
        developer.log(
          'POST $postRoute | failure status=${response.statusCode} | body=${truncateForLog(response.body)}',
          name: _logName,
          level: 1000,
        );
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
          developer.log(
            'POST $postRoute | ignored key=$key not mapped to enum',
            name: _logName,
          );
        }
      });

      developer.log(
        'POST $postRoute | decodedDimensions=${result.length}',
        name: _logName,
      );
      return result;
    } catch (e, stack) {
      stopwatch.stop();
      developer.log(
        'POST $postRoute | exception=$e | duration=${stopwatch.elapsedMilliseconds}ms',
        name: _logName,
        error: e,
        stackTrace: stack,
        level: 1000,
      );
      throw Exception('An error occurred during the distributions request: $e');
    }
  }
}
