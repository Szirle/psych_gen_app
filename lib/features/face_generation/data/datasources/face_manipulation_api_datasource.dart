import 'preview_image_codec.dart';
export 'preview_image_codec.dart' show decodeImageGridForDisplay;
import 'dart:convert';
import 'dart:developer' as developer;
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:psych_gen_app/core/constants/api_config.dart';
import 'package:psych_gen_app/core/utils/logging.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/face_manipulation_request.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/full_resolution_face.dart';

class FaceManipulationApiDataSource {
  static const String _logName = 'FaceManipulationApiDataSource';
  static String get postRoute => ApiConfig.resolve('/images');
  static String get fullResolutionRoute => ApiConfig.resolve('/images/full');
  static String get fullResolutionStatusRoute =>
      ApiConfig.resolve('/images/full/status');
  static String get fullResolutionPrioritizeRoute =>
      ApiConfig.resolve('/images/full/prioritize');
  static String get fullResolutionCancelRoute =>
      ApiConfig.resolve('/images/full/cancel');

  Future<void> cancelFullResolutionGeneration(int previewRevision) async {
    final response = await http.post(
      Uri.parse(fullResolutionCancelRoute),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'preview_revision': previewRevision}),
    );
    if (response.statusCode != 200) {
      throw Exception(
        'Full-resolution cancellation failed (${response.statusCode}).',
      );
    }
  }

  Future<FullResolutionResponse> postFullResolutionFace(
    FaceManipulationRequest requestBody,
    List<int> selectedLevels,
    Uint8List previewBytes,
  ) async {
    final payload = requestBody.toJson()
      ..['change_face'] = false
      ..['selected_levels'] = selectedLevels
      ..['preview_image'] = base64Encode(previewBytes);
    final response = await http.post(
      Uri.parse('$fullResolutionRoute?format=webp&quality=95'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode(payload),
    );
    if (response.statusCode != 200) {
      String message = response.body;
      try {
        final dynamic body = json.decode(response.body);
        if (body is Map && body['error'] is String) {
          message = body['error'] as String;
        }
      } catch (_) {}
      throw Exception('Request failed (${response.statusCode}): $message');
    }
    final dynamic body = json.decode(response.body);
    return _decodeFullResolutionResponse(body);
  }

  Future<FullResolutionBatchUpdate> fetchFullResolutionUpdates(
    String cacheKey,
    int after,
  ) async {
    final response = await http
        .post(
          Uri.parse(fullResolutionStatusRoute),
          headers: {'Content-Type': 'application/json'},
          body: json.encode({
            'cache_key': cacheKey,
            'after': after,
            'wait_ms': 10000,
          }),
        )
        .timeout(const Duration(seconds: 15));
    if (response.statusCode != 200) {
      throw Exception(
          'Full-resolution status failed (${response.statusCode}).');
    }
    final dynamic body = json.decode(response.body);
    if (body is! Map ||
        body['items'] is! List ||
        body['cursor'] is! int ||
        body['complete'] is! bool) {
      throw const FormatException('Invalid full-resolution status response.');
    }
    return FullResolutionBatchUpdate(
      items: (body['items'] as List)
          .map<FullResolutionResponse>(_decodeFullResolutionResponse)
          .toList(),
      cursor: body['cursor'] as int,
      complete: body['complete'] as bool,
      error: body['error'] as String?,
    );
  }

  Future<void> prioritizeFullResolution(
    String cacheKey,
    List<int> selectedLevels,
  ) async {
    final response = await http.post(
      Uri.parse(fullResolutionPrioritizeRoute),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({
        'cache_key': cacheKey,
        'selected_levels': selectedLevels,
      }),
    );
    if (response.statusCode != 200) {
      throw Exception(
        'Full-resolution prioritization failed (${response.statusCode}).',
      );
    }
  }

  FullResolutionResponse _decodeFullResolutionResponse(dynamic body) {
    if (body is! Map ||
        body['image'] is! String ||
        body['latent_npy'] is! String ||
        body['latent_shape'] is! List ||
        (body['latent_shape'] as List).length != 2 ||
        body['latent_shape'][0] != 18 ||
        body['latent_shape'][1] != 512 ||
        body['selected_levels'] is! List ||
        body['cache_key'] is! String ||
        body['preview_is_low_resolution'] is! bool ||
        body['resolution'] != 1024) {
      throw const FormatException('Invalid full-resolution image response.');
    }
    return FullResolutionResponse(
      face: FullResolutionFace(
        imageBytes: base64Decode(body['image'] as String),
        latentNpyBytes: base64Decode(body['latent_npy'] as String),
      ),
      selectedLevels: (body['selected_levels'] as List)
          .map<int>((item) => item as int)
          .toList(),
      cacheKey: body['cache_key'] as String,
      previewIsLowResolution: body['preview_is_low_resolution'] as bool,
    );
  }

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
