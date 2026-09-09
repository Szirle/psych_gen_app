import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:psych_gen_app/core/constants/api_config.dart';
import '../../domain/entities/stimuli_selection_request.dart';
import 'preview_image_codec.dart';

class StimuliSelectionApiDataSource {
  final http.Client _client;
  StimuliSelectionApiDataSource({http.Client? client})
      : _client = client ?? http.Client();

  Future<StimuliSelectionPreview> fetchPreview(
      StimuliSelectionRequest request) async {
    final response = await _client.post(
      Uri.parse(ApiConfig.resolve('/stimuli/preview')),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(request.toJson()),
    );
    if (response.statusCode != 200) {
      String message =
          'Stimuli preview request failed (${response.statusCode}).';
      try {
        final body = jsonDecode(response.body);
        if (body is Map && body['error'] is String) message = body['error'];
      } catch (_) {}
      throw Exception(message);
    }
    final body = jsonDecode(response.body);
    if (body is! Map ||
        body['preview_revision'] != request.previewRevision ||
        body['images'] is! List ||
        body['sampling'] is! Map) {
      throw const FormatException('Invalid stimuli preview response.');
    }
    final sampling = body['sampling'] as Map;
    final source = sampling['source'];
    final count = sampling['eligible_count'];
    if (!['generator', 'ratings'].contains(source) ||
        (source == 'ratings' && (count is! int || count < 1)) ||
        (source == 'generator' && count != null) ||
        sampling['sampled_with_replacement'] is! bool) {
      throw const FormatException('Invalid sampling metadata.');
    }
    return StimuliSelectionPreview(
      images: decodeImageGridForDisplay(body['images'], const [3, 3]),
      previewRevision: request.previewRevision,
      source: source as String,
      eligibleCount: count as int?,
      sampledWithReplacement: sampling['sampled_with_replacement'] as bool,
    );
  }

  void close() => _client.close();
}
