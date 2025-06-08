import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:psych_gen_app/api/api_service.dart';
import 'package:psych_gen_app/model/face_manipulation_request.dart';
import 'package:psych_gen_app/repository/face_manipulation_repository.dart';

class FaceManipulationRepositoryImpl implements FaceManipulationRepository {
  final ApiService _apiService;

  FaceManipulationRepositoryImpl({ApiService? apiService})
      : _apiService = apiService ?? ApiService();

  @override
  Future<List<Uint8List>> getFaceImages(FaceManipulationRequest request) async {
    if (kDebugMode) {
      return _generateMockImages(request);
    } else {
      return _apiService.postFaceManipulation(request);
    }
  }

  Future<List<Uint8List>> _generateMockImages(
      FaceManipulationRequest request) async {
    // Calculate total number of images needed (product of all nLevels)
    int totalImages = 1;
    for (var dimension in request.manipulatedDimensions) {
      totalImages *= dimension.nLevels;
    }

    List<Uint8List> images = [];

    for (int i = 0; i < totalImages; i++) {
      final bytes = await _generateIconImage(Icons.person, Colors.blueGrey);
      images.add(bytes);
    }

    return images;
  }

  Future<Uint8List> _generateIconImage(IconData icon, Color color) async {
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    const size = Size(200, 200);

    // Draw background
    final paint = Paint()..color = Colors.grey[100]!;
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), paint);

    // Draw icon
    final textPainter = TextPainter(
      text: TextSpan(
        text: String.fromCharCode(icon.codePoint),
        style: TextStyle(
          fontSize: 80,
          fontFamily: icon.fontFamily,
          color: color,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(
        (size.width - textPainter.width) / 2,
        (size.height - textPainter.height) / 2,
      ),
    );

    final picture = recorder.endRecording();
    final img = await picture.toImage(size.width.toInt(), size.height.toInt());
    final byteData = await img.toByteData(format: ui.ImageByteFormat.png);
    return byteData!.buffer.asUint8List();
  }
}
