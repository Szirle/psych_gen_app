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
  Future<List<List<Uint8List?>>> getFaceImages(
      FaceManipulationRequest request) async {
    if (kDebugMode) {
      return _generateMockImages(request);
    } else {
      return _apiService.postFaceManipulation(request);
    }
  }

  Future<List<List<Uint8List?>>> _generateMockImages(
      FaceManipulationRequest request) async {
    // For mock data, generate a simple 2D grid based on dimensions
    if (request.manipulatedDimensions.isEmpty) {
      return [];
    }

    if (request.manipulatedDimensions.length == 1) {
      // Single dimension - one row
      final dim = request.manipulatedDimensions[0];
      List<Uint8List?> row = [];
      for (int i = 0; i < dim.nLevels; i++) {
        final bytes = await _generateIconImage(Icons.person, Colors.blueGrey);
        row.add(bytes);
      }
      return [row];
    } else if (request.manipulatedDimensions.length == 2) {
      // Two dimensions - full grid
      final dim1 = request.manipulatedDimensions[0];
      final dim2 = request.manipulatedDimensions[1];

      List<List<Uint8List?>> grid = [];
      for (int row = 0; row < dim2.nLevels; row++) {
        List<Uint8List?> rowImages = [];
        for (int col = 0; col < dim1.nLevels; col++) {
          final bytes = await _generateIconImage(Icons.person, Colors.blueGrey);
          rowImages.add(bytes);
        }
        grid.add(rowImages);
      }
      return grid;
    } else {
      // Three dimensions - for now just return a grid for the first slice
      final dim1 = request.manipulatedDimensions[0];
      final dim2 = request.manipulatedDimensions[1];

      List<List<Uint8List?>> grid = [];
      for (int row = 0; row < dim2.nLevels; row++) {
        List<Uint8List?> rowImages = [];
        for (int col = 0; col < dim1.nLevels; col++) {
          final bytes = await _generateIconImage(Icons.person, Colors.blueGrey);
          rowImages.add(bytes);
        }
        grid.add(rowImages);
      }
      return grid;
    }
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
